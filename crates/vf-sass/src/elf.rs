//! The container of machine code (a cubin: an ELF64 file) read into sections and written back.
//!
//! Written back, a file whose sections were not changed is the original byte for byte.
//! Sections whose size changed move what follows them. This is a faithful port of
//! `reference/elf.py`.

use crate::fast::FastMap;

const SHT_NULL: u32 = 0;
const SHT_SYMTAB: u32 = 2;
const SHT_NOBITS: u32 = 8;
const PT_LOAD: u32 = 1;
const PT_PHDR: u32 = 6;
/// No section of a cubin asks for more (the largest seen is 128): a file that does is not one.
const MAX_ALIGN: u64 = 1 << 16;

/// One section of the ELF file.
#[derive(Clone)]
pub struct Section {
    pub name: String,
    pub name_offset: u32,
    pub stype: u32,
    pub flags: u64,
    pub addr: u64,
    pub offset: u64,
    pub size: u64,
    pub link: u32,
    pub info: u32,
    pub align: u64,
    pub entsize: u64,
    pub data: Vec<u8>,
}

/// A program header (segment).
#[derive(Clone)]
pub struct Segment {
    pub stype: u32,
    pub flags: u32,
    pub offset: u64,
    pub vaddr: u64,
    pub paddr: u64,
    pub filesz: u64,
    pub memsz: u64,
    pub align: u64,
}

/// A symbol from the symbol table.
pub struct Symbol {
    pub name: String,
    pub name_offset: u32,
    pub info: u8,
    pub other: u8,
    pub shndx: u16,
    pub value: u64,
    pub size: u64,
}

/// The ELF container, holding sections and segments.
#[allow(dead_code)]
pub struct Elf {
    pub header: Vec<u8>,
    phoff: u64,
    shoff: u64,
    phentsize: u16,
    phnum: u16,
    shentsize: u16,
    shnum: u16,
    shstrndx: u16,
    pub sections: Vec<Section>,
    segments: Vec<Segment>,
}

/// The string that starts at `offset` in a string table.
fn string_at(table: &[u8], offset: u32) -> Result<String, String> {
    let rest = table.get(offset as usize..).unwrap_or(&[]);
    let Some(length) = rest.iter().position(|&b| b == 0) else {
        return Err("a name that does not end in its table".into());
    };
    String::from_utf8(rest[..length].to_vec()).map_err(|_| "a name that is not text".into())
}

fn read_u16(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(data[off..off + 2].try_into().unwrap())
}

fn read_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

fn read_u64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}

#[allow(dead_code)]
fn write_u16(data: &mut [u8], off: usize, v: u16) {
    data[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

#[allow(dead_code)]
fn write_u32(data: &mut [u8], off: usize, v: u32) {
    data[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn write_u64(data: &mut [u8], off: usize, v: u64) {
    data[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

impl Elf {
    /// Parse an ELF file from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 64 || data[..4] != *b"\x7fELF" || data[4] != 2 || data[5] != 1 {
            return Err("not a 64-bit little-endian ELF".into());
        }
        let header = data[..64].to_vec();
        let phoff = read_u64(data, 0x20);
        let shoff = read_u64(data, 0x28);
        let phentsize = read_u16(data, 0x36);
        let phnum = read_u16(data, 0x38);
        let shentsize = read_u16(data, 0x3a);
        let shnum = read_u16(data, 0x3c);
        let shstrndx = read_u16(data, 0x3e);

        if shentsize != 64 || (phnum != 0 && phentsize != 56) {
            return Err("unexpected ELF header sizes".into());
        }

        let within = |offset: u64, size: u64| offset.checked_add(size).is_some_and(|end| end <= data.len() as u64);
        if !within(shoff, 64 * shnum as u64) || !within(phoff, 56 * phnum as u64) {
            return Err("the section or program headers are outside the file".into());
        }

        let mut sections = Vec::with_capacity(shnum as usize);
        for i in 0..shnum as usize {
            let off = shoff as usize + 64 * i;
            let name_offset = read_u32(data, off);
            let stype = read_u32(data, off + 4);
            let flags = read_u64(data, off + 8);
            let addr = read_u64(data, off + 16);
            let offset = read_u64(data, off + 24);
            let size = read_u64(data, off + 32);
            let link = read_u32(data, off + 40);
            let info = read_u32(data, off + 44);
            let align = read_u64(data, off + 48);
            let entsize = read_u64(data, off + 56);
            let section_data = if stype == SHT_NULL || stype == SHT_NOBITS {
                Vec::new()
            } else if within(offset, size) {
                data[offset as usize..(offset as usize + size as usize)].to_vec()
            } else {
                return Err(format!("section {i} is outside the file"));
            };
            if align > MAX_ALIGN {
                return Err(format!("section {i} asks for an alignment of {align}"));
            }
            sections.push(Section {
                name: String::new(), // filled below
                name_offset,
                stype,
                flags,
                addr,
                offset,
                size,
                link,
                info,
                align,
                entsize,
                data: section_data,
            });
        }

        // Read section names (clone to avoid borrow conflict)
        let names = match sections.get(shstrndx as usize) {
            Some(s) => s.data.clone(),
            None => return Err("no table of section names".into()),
        };
        for s in &mut sections {
            s.name = string_at(&names, s.name_offset)?;
        }

        let mut segments = Vec::with_capacity(phnum as usize);
        for i in 0..phnum as usize {
            let off = phoff as usize + 56 * i;
            segments.push(Segment {
                stype: read_u32(data, off),
                flags: read_u32(data, off + 4),
                offset: read_u64(data, off + 8),
                vaddr: read_u64(data, off + 16),
                paddr: read_u64(data, off + 24),
                filesz: read_u64(data, off + 32),
                memsz: read_u64(data, off + 40),
                align: read_u64(data, off + 48),
            });
        }

        Ok(Elf {
            header,
            phoff,
            shoff,
            phentsize,
            phnum,
            shentsize,
            shnum,
            shstrndx,
            sections,
            segments,
        })
    }

    /// Find a section by name.
    pub fn section(&self, name: &str) -> Option<usize> {
        self.sections.iter().position(|s| s.name == name)
    }

    /// Map each section name to its first section index.
    #[allow(dead_code)]
    pub(crate) fn section_index_map(&self) -> FastMap<&str, usize> {
        let mut map = FastMap::with_capacity_and_hasher(self.sections.len(), Default::default());
        for (i, s) in self.sections.iter().enumerate() {
            map.entry(s.name.as_str()).or_insert(i);
        }
        map
    }

    /// Read the symbol table.
    pub fn symbols(&self) -> Result<Vec<Symbol>, String> {
        let Some(table) = self.sections.iter().find(|s| s.stype == SHT_SYMTAB) else {
            return Err("no table of symbols".into());
        };
        let Some(strings) = self.sections.get(table.link as usize).map(|s| s.data.as_slice()) else {
            return Err("no table of symbol names".into());
        };
        if !table.data.len().is_multiple_of(24) {
            return Err("a table of symbols cut short".into());
        }
        let mut out = Vec::new();
        let mut pos = 0;
        while pos + 24 <= table.data.len() {
            let name_off = read_u32(&table.data, pos);
            let info = table.data[pos + 4];
            let other = table.data[pos + 5];
            let shndx = read_u16(&table.data, pos + 6);
            let value = read_u64(&table.data, pos + 8);
            let size = read_u64(&table.data, pos + 16);
            out.push(Symbol {
                name: string_at(strings, name_off)?,
                name_offset: name_off,
                info,
                other,
                shndx,
                value,
                size,
            });
            pos += 24;
        }
        Ok(out)
    }

    /// Write the symbol table back.
    pub fn set_symbols(&mut self, symbols: &[Symbol]) {
        let Some(table_idx) = self.sections.iter().position(|s| s.stype == SHT_SYMTAB) else {
            return;
        };
        let mut data = Vec::with_capacity(symbols.len() * 24);
        for s in symbols {
            data.extend_from_slice(&s.name_offset.to_le_bytes());
            data.push(s.info);
            data.push(s.other);
            data.extend_from_slice(&s.shndx.to_le_bytes());
            data.extend_from_slice(&s.value.to_le_bytes());
            data.extend_from_slice(&s.size.to_le_bytes());
        }
        self.sections[table_idx].data = data;
    }

    /// Write the ELF file back to bytes.
    pub fn write(&self) -> Vec<u8> {
        let mut out = self.header.clone();
        let mut offsets = Vec::with_capacity(self.sections.len());

        for s in &self.sections {
            if s.stype == SHT_NULL {
                offsets.push(0u64);
                continue;
            }
            let align = s.align.max(1) as usize;
            let pad = (align - (out.len() % align)) % align;
            out.extend(std::iter::repeat_n(0u8, pad));
            offsets.push(out.len() as u64);
            if s.stype != SHT_NOBITS {
                out.extend_from_slice(&s.data);
            }
        }

        // Align to 8 bytes before section headers
        let pad = (8 - (out.len() % 8)) % 8;
        out.extend(std::iter::repeat_n(0u8, pad));
        let shoff = out.len() as u64;

        // Section headers
        for (s, &offset) in self.sections.iter().zip(offsets.iter()) {
            let size = if s.stype == SHT_NULL || s.stype == SHT_NOBITS {
                s.size
            } else {
                s.data.len() as u64
            };
            out.extend_from_slice(&s.name_offset.to_le_bytes());
            out.extend_from_slice(&s.stype.to_le_bytes());
            out.extend_from_slice(&s.flags.to_le_bytes());
            out.extend_from_slice(&s.addr.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
            out.extend_from_slice(&s.link.to_le_bytes());
            out.extend_from_slice(&s.info.to_le_bytes());
            out.extend_from_slice(&s.align.to_le_bytes());
            out.extend_from_slice(&s.entsize.to_le_bytes());
        }

        // Program headers
        let phoff = if self.phnum > 0 { out.len() as u64 } else { 0 };
        for seg in &self.segments {
            let (offset, filesz, memsz) = if seg.stype == PT_PHDR || (seg.stype == PT_LOAD && seg.offset == self.phoff)
            {
                (phoff, 56 * self.phnum as u64, 56 * self.phnum as u64)
            } else if seg.filesz > 0 || seg.memsz > 0 {
                // Find sections inside this segment and compute new bounds
                let inside: Vec<usize> = self
                    .sections
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| {
                        s.stype != SHT_NULL
                            && s.offset >= seg.offset
                            && s.offset.saturating_add(section_length(s)) <= seg.offset.saturating_add(seg.filesz)
                    })
                    .map(|(i, _)| i)
                    .collect();
                if inside.is_empty() {
                    (seg.offset, seg.filesz, seg.memsz)
                } else {
                    let start = offsets[inside[0]];
                    let end = inside
                        .iter()
                        .map(|&i| {
                            offsets[i]
                                + if self.sections[i].stype == SHT_NOBITS {
                                    0
                                } else {
                                    self.sections[i].data.len() as u64
                                }
                        })
                        .max()
                        .unwrap();
                    (
                        start,
                        end - start,
                        seg.memsz.wrapping_sub(seg.filesz).wrapping_add(end - start),
                    )
                }
            } else {
                (seg.offset, seg.filesz, seg.memsz)
            };
            out.extend_from_slice(&seg.stype.to_le_bytes());
            out.extend_from_slice(&seg.flags.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&seg.vaddr.to_le_bytes());
            out.extend_from_slice(&seg.paddr.to_le_bytes());
            out.extend_from_slice(&filesz.to_le_bytes());
            out.extend_from_slice(&memsz.to_le_bytes());
            out.extend_from_slice(&seg.align.to_le_bytes());
        }

        // Patch the header with new offsets
        write_u64(&mut out, 0x20, phoff);
        write_u64(&mut out, 0x28, shoff);

        out
    }
}

fn section_length(s: &Section) -> u64 {
    if s.stype == SHT_NOBITS { 0 } else { s.size }
}
