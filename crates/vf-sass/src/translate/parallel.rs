//! Bounded worker-thread execution for SASS cubin translation (`VF_THREADS`),
//! modeled on `vf_ptx::parallel` with calling-thread fallback on thread creation failure.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_MAX_THREADS: usize = 8;

fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .max(1)
}

pub(super) fn effective_threads_with_env(work_items: usize, large_enough: bool, raw: Option<&str>) -> usize {
    if !large_enough || work_items <= 1 {
        return 1;
    }
    let avail = available_threads();
    let cap = match raw.map(str::trim).filter(|s| !s.is_empty()) {
        None => avail.min(DEFAULT_MAX_THREADS),
        Some(val) => match val.parse::<usize>() {
            Ok(0) | Ok(1) => 1,
            Ok(n) => n.min(avail),
            Err(_) => avail.min(DEFAULT_MAX_THREADS),
        },
    };
    cap.min(work_items).max(1)
}

pub(super) fn effective_threads(work_items: usize, large_enough: bool) -> usize {
    let raw = std::env::var("VF_THREADS").ok();
    effective_threads_with_env(work_items, large_enough, raw.as_deref())
}

/// Execute `f` over `0..items.len()` using up to `num_threads` threads (calling thread +
/// up to `num_threads - 1` workers named `vf-sass-N`). Each thread creates its own `S` once via
/// `init` and reuses it across work items. Returns results in `0..items.len()` order, and the
/// earliest-index error if any item fails.
pub(super) fn par_map_init<T, S, R, E, Init, F>(items: &[T], num_threads: usize, init: Init, f: F) -> Result<Vec<R>, E>
where
    T: Sync,
    R: Send,
    E: Send,
    Init: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &T) -> Result<R, E> + Sync,
{
    par_map_with_spawner(items, num_threads, &init, &f, |builder, scope, run| {
        builder.spawn_scoped(scope, run).is_ok()
    })
}

pub(super) fn par_map_with_spawner<T, S, R, E, Init, F, Spawn>(
    items: &[T],
    num_threads: usize,
    init: &Init,
    f: &F,
    mut spawn_fn: Spawn,
) -> Result<Vec<R>, E>
where
    T: Sync,
    R: Send,
    E: Send,
    Init: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &T) -> Result<R, E> + Sync,
    Spawn:
        for<'s, 'e> FnMut(std::thread::Builder, &'s std::thread::Scope<'s, 'e>, Box<dyn FnOnce() + Send + 's>) -> bool,
{
    if num_threads <= 1 || items.len() <= 1 {
        let mut state = init();
        let mut out = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            out.push(f(&mut state, i, item)?);
        }
        return Ok(out);
    }

    let slots: Vec<Mutex<Option<Result<R, E>>>> = (0..items.len()).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let first_err = AtomicUsize::new(usize::MAX);

    let run_worker = || {
        let mut state = init();
        loop {
            let idx = next.fetch_add(1, Ordering::Relaxed);
            if idx >= items.len() || idx >= first_err.load(Ordering::Relaxed) {
                break;
            }
            let res = f(&mut state, idx, &items[idx]);
            if res.is_err() {
                first_err.fetch_min(idx, Ordering::Relaxed);
            }
            *slots[idx].lock().unwrap() = Some(res);
        }
    };

    std::thread::scope(|scope| {
        for n in 1..num_threads {
            let builder = std::thread::Builder::new().name(format!("vf-sass-{n}"));
            let worker: Box<dyn FnOnce() + Send + '_> = Box::new(&run_worker);
            if !spawn_fn(builder, scope, worker) {
                break;
            }
        }
        run_worker();
    });

    let mut out = Vec::with_capacity(items.len());
    for slot in slots {
        out.push(slot.into_inner().unwrap().expect("every work item must complete")?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_threads_respects_env_and_small_gate() {
        assert_eq!(effective_threads_with_env(10, false, Some("4")), 1);
        assert_eq!(effective_threads_with_env(1, true, Some("4")), 1);
        assert_eq!(effective_threads_with_env(10, true, Some("0")), 1);
        assert_eq!(effective_threads_with_env(10, true, Some("1")), 1);
        let max = available_threads().min(DEFAULT_MAX_THREADS);
        assert_eq!(effective_threads_with_env(100, true, None), max);
        assert_eq!(effective_threads_with_env(100, true, Some("bogus")), max);
    }

    #[test]
    fn par_map_fallback_on_spawn_failure_and_deterministic_error() {
        let data: Vec<usize> = (0..12).collect();
        let res = par_map_with_spawner(
            &data,
            4,
            &|| 10usize,
            &|state, _i, &x| Ok::<_, usize>(*state + x),
            |_, _, _| false,
        )
        .unwrap();
        assert_eq!(res, (10..22).collect::<Vec<_>>());

        let err_res = par_map_with_spawner(
            &data,
            4,
            &|| (),
            &|_, i, _| if i == 3 || i == 7 { Err(i) } else { Ok(i) },
            |builder, scope, run| builder.spawn_scoped(scope, run).is_ok(),
        );
        assert_eq!(err_res, Err(3));
    }
}
