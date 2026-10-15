//! v3 FDJ front-end: fetch-scan -> mine -> decode (VMP 3.x).
use anyhow::Result;
use crate::frontend::{FetchHit, VmFrontend};
use crate::frontend::fetch_finder::{scan_fetch_sites_strict, scan_fetch_sites_strict_bits, FetchSite};
use crate::pe_loader::{BinFmt, PEBinary};

pub struct V3Fdj;

fn vmp_section_data<'a>(binary: &'a PEBinary) -> Result<(Vec<u8>, u64)> {
    let pe = binary.parse_manual()?;
    let mut best: Option<(Vec<u8>, u64)> = None;
    for s in &pe.sections {
        if [".text", ".rdata", ".data", ".pdata", ".reloc", ".rsrc", ".buildid"].contains(&s.name_lossy.as_str()) {
            continue;
        }
        let sz = s.raw_size as usize;
        if sz == 0 {
            continue;
        }
        let off = s.raw_ptr as usize;
        let end = off.saturating_add(sz).min(binary.data.len());
        if off >= end {
            continue;
        }
        let cand = (binary.data[off..end].to_vec(), pe.image_base + s.rva as u64);
        if best.as_ref().map(|(d, _)| d.len()).unwrap_or(0) < cand.0.len() {
            best = Some(cand);
        }
    }
    best.ok_or_else(|| anyhow::anyhow!("no VMP data section"))
}

/// ELF exec sections that are toolchain stubs, never VM containers.
fn is_standard_elf_exec(name: &str) -> bool {
    matches!(name, ".init" | ".fini" | ".plt" | ".plt.got" | ".plt.sec" | ".text")
}

/// Max |disp| for fetch `[reg+disp]` on 32-bit ELF (`FETCH_MAX_DISP`,
/// default 0 = historic disp-free rule; VMP x86 fetches `[edi-1]`).
fn fetch_max_disp() -> i64 {
    std::env::var("FETCH_MAX_DISP").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// ELF strict fetch sites over every executable region, decoded at the
/// ELF class bitness. VMP-on-ELF may virtualize into `.text` itself, so
/// all exec regions are scanned once detection has a VM marker.
fn elf_fetch_sites(binary: &PEBinary) -> Result<Vec<FetchSite>> {
    let bits = binary.decode_bits();
    let max_disp = fetch_max_disp();
    let mut out = Vec::new();
    for (_, va, data) in binary.elf_exec_regions()? {
        out.extend(scan_fetch_sites_strict_bits(&data, va, bits, max_disp));
    }
    Ok(out)
}

/// ELF VM marker: a non-standard exec section (`.vmp*`, renamed or
/// mutated), or a section-stripped image (packed loader, `loadN`).
fn elf_has_vm_region(binary: &PEBinary) -> bool {
    binary.elf_exec_regions()
        .map(|r| r.iter().any(|(n, _, _)| !is_standard_elf_exec(n)))
        .unwrap_or(false)
}

impl VmFrontend for V3Fdj {
    fn name(&self) -> &'static str {
        "vmp3-fdj"
    }
    fn detect(&self, binary: &PEBinary) -> Result<bool> {
        // VMP 3.x: large non-standard section + strict fetch sites present.
        if binary.fmt() == BinFmt::Elf {
            return Ok(elf_has_vm_region(binary) && !elf_fetch_sites(binary)?.is_empty());
        }
        let (data, base) = match vmp_section_data(binary) {
            Ok(v) => v,
            Err(_) => return Ok(false),
        };
        Ok(!scan_fetch_sites_strict(&data, base).is_empty())
    }
    fn fetch_stream(&self, _binary: &PEBinary, _trace: Option<&[u64]>) -> Result<Vec<FetchHit>> {
        // Static-only v3 decode needs runtime keys; with a trace, hits are
        // decoded per-site by the caller. Without one, report no hits rather
        // than guessing (see harness::snapshot for capture).
        Ok(Vec::new())
    }
    fn handler_addrs(&self, binary: &PEBinary, _hits: &[FetchHit]) -> Result<Vec<u64>> {
        if binary.fmt() == BinFmt::Elf {
            return Ok(elf_fetch_sites(binary)?.into_iter().map(|s| s.va).collect());
        }
        let (data, base) = vmp_section_data(binary)?;
        Ok(scan_fetch_sites_strict(&data, base).into_iter().map(|s: FetchSite| s.va).collect())
    }
}
