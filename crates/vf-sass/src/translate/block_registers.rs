//! GV100 register limit for an entire CTA. A graph records its kernel before
//! the driver checks this limit, so a launch-time retry can arrive too late. A
//! kernel that requires its block size is launched with no other: a program may
//! read the limit before any launch (Triton does) and stop there.

/// Maximum registers per thread consistent with every previously launchable
/// CTA under the function's register claim and declared block dimensions.
pub(super) fn budget(registers: u32, info: &[u8]) -> u32 {
    // GV100's four subpartitions each own 16,384 registers and allocate
    // a warp's 32 registers per thread in units of 256 registers.
    let regs_per_warp = (32 * registers.max(1)).next_multiple_of(256);
    let hardware_threads = (16_384 / regs_per_warp * 4 * 32).min(1024);
    let threads = declared_threads(info).unwrap_or(1024).min(hardware_threads);
    registers_for(threads)
}

/// The most registers a thread may have for a block of `threads` threads to be launched on a
/// GV100.
/// Effective resident warps per GV100 subpartition for `registers`, rounding down to whole CTAs
/// when `info` fixes a CTA size (`EIATTR_REQNTID`).
pub(super) fn occupancy_warps(registers: u32, info: &[u8]) -> u32 {
    let regs_per_warp = (32 * registers.max(1)).next_multiple_of(256);
    let per_sub = (16_384 / regs_per_warp).min(16);
    if let Some(req) = required_threads(info) {
        let groups = req.div_ceil(128).max(1);
        (per_sub / groups) * groups
    } else {
        per_sub
    }
}

/// Largest register count per thread (`<= 255`) preserving at least `occupancy_warps(registers, info)`
/// on GV100.
pub(super) fn occupancy_target(registers: u32, info: &[u8]) -> u32 {
    let w = occupancy_warps(registers, info);
    16_384u32.checked_div(w).map_or(255, |n| (n / 256 * 8).min(255))
}

pub(super) fn registers_for(threads: u32) -> u32 {
    // The per-block admission check rounds warp count up to four
    // subpartitions; each 128-thread group consumes the same allocation.
    let groups = threads.div_ceil(128).max(1);
    (65_536 / (128 * groups) / 8 * 8).min(255)
}

/// The block size an entry requires (EIATTR_REQNTID, which Triton writes for every kernel and
/// CUDA C never): the driver launches it with that block and no other. `None` when it requires
/// none, or one that is not understood or that no GPU launches (more than 1,024 threads).
pub(super) fn required_threads(mut info: &[u8]) -> Option<u32> {
    let mut required: Option<u32> = None;
    while !info.is_empty() {
        let [format, kind, lo, hi] = *info.first_chunk::<4>()?;
        let size = if format == 4 {
            u16::from_le_bytes([lo, hi]) as usize
        } else {
            0
        };
        let (payload, rest) = info.get(4..)?.split_at_checked(size)?;
        if kind == 0x10 {
            if format != 4 || size != 12 {
                return None;
            }
            let dims = payload.as_chunks::<4>().0.iter().copied().map(u32::from_le_bytes);
            let n = dims.into_iter().try_fold(1u32, u32::checked_mul)?;
            if n == 0 || n > 1024 {
                return None;
            }
            required = Some(required.map_or(n, |r| r.min(n)));
        }
        info = rest;
    }
    required
}

/// An absent or unrecognized shape never narrows the hardware bound.
fn declared_threads(mut info: &[u8]) -> Option<u32> {
    let mut maximum = 1024;
    while !info.is_empty() {
        let [format, kind, lo, hi] = *info.first_chunk::<4>()?;
        let size = if format == 4 {
            u16::from_le_bytes([lo, hi]) as usize
        } else {
            0
        };
        let (payload, rest) = info.get(4..)?.split_at_checked(size)?;
        if matches!(kind, 0x05 | 0x10) {
            // EIATTR_MAX_THREADS / EIATTR_REQNTID
            if format != 4 || size != 12 {
                return None;
            }
            let dims = payload.as_chunks::<4>().0.iter().copied().map(u32::from_le_bytes);
            let n = dims.into_iter().try_fold(1u32, u32::checked_mul)?;
            if n == 0 {
                return None;
            }
            maximum = maximum.min(n);
        }
        info = rest;
    }
    Some(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shape(kind: u8, dims: [u32; 3]) -> Vec<u8> {
        let mut info = vec![4, kind, 12, 0];
        for d in dims {
            info.extend_from_slice(&d.to_le_bytes());
        }
        info
    }
    #[test]
    fn register_thresholds_include_whole_warps_and_scheduler_partitions() {
        assert_eq!(budget(12, &[]), 64);
        assert_eq!(budget(64, &[]), 64);
        assert_eq!(budget(65, &[]), 72);
        assert_eq!(budget(80, &[]), 80);
        assert_eq!(budget(129, &[]), 168);
        assert_eq!(budget(255, &[]), 255);
        for src in 1..=255 {
            for threads in 1u32..=1024 {
                let allowed = |r: u32| threads.div_ceil(128) * 4 * (r * 32).next_multiple_of(256) <= 65536;
                assert_eq!(allowed(src), allowed(budget(src, &[])));
            }
        }
    }
    #[test]
    fn small_shapes_keep_the_plain_first_image() {
        for kind in [0x05, 0x10] {
            assert_eq!(budget(64, &shape(kind, [32, 4, 2])), 255);
            assert_eq!(budget(64, &shape(kind, [512, 1, 1])), 128);
            assert_eq!(budget(64, &shape(kind, [897, 1, 1])), 64);
        }
        assert_eq!(budget(64, &[4, 0x10, 12, 0, 1, 0]), 64);
        assert_eq!(budget(64, &shape(0x05, [0, 1, 1])), 64);
        assert_eq!(budget(64, &shape(0x10, [u32::MAX, 4, 4])), 64);
    }
    #[test]
    fn a_required_block_is_read_from_its_own_attribute_alone() {
        assert_eq!(required_threads(&[]), None);
        assert_eq!(required_threads(&shape(0x05, [512, 1, 1])), None);
        assert_eq!(required_threads(&shape(0x10, [512, 1, 1])), Some(512));
        assert_eq!(required_threads(&shape(0x10, [32, 4, 2])), Some(256));
        let mut both = shape(0x05, [1024, 1, 1]);
        both.extend(shape(0x10, [128, 1, 1]));
        assert_eq!(required_threads(&both), Some(128));
        assert_eq!(required_threads(&shape(0x10, [0, 1, 1])), None);
        assert_eq!(required_threads(&shape(0x10, [1025, 1, 1])), None);
        assert_eq!(required_threads(&shape(0x10, [u32::MAX, 4, 4])), None);
        assert_eq!(required_threads(&[4, 0x10, 12, 0, 1, 0]), None);
        assert_eq!(
            (registers_for(512), registers_for(1024), registers_for(384)),
            (128, 64, 168)
        );
        assert_eq!(
            (registers_for(32), registers_for(129), registers_for(896)),
            (255, 255, 72)
        );
    }
}
