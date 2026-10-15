//! Fixture tests: synthetic section-less ELF32 (packed-loader shape) with
//! a hand-built x86-32 fetch chain. No external files.
use vmp_devirt::backend::value_cryptor::CryptOp;
use vmp_devirt::frontend::cryptor_miner::mine_cryptor;
use vmp_devirt::frontend::emu_miner::{image_of, mine_cryptor_emu};
use vmp_devirt::frontend::fetch_finder::{scan_fetch_sites_strict_bits, FetchSite};
use vmp_devirt::pe_loader::{BinFmt, PEBinary};
use iced_x86::Register;

const VA: u64 = 0x8049100;
const FETCH: u64 = VA + 5;
/// File-backed code size (handler + int3 padding; miners read ahead).
const LEN: u64 = 0x100;

/// Handler: junk constant before the fetch, carry-dependent op, junk-reg
/// xor, rotate, rolling-key update. 32-bit encodings.
fn handler() -> Vec<u8> {
    vec![
        0xB9, 0x55, 0x00, 0x00, 0x00, // mov ecx, 0x55        (junk const, pre-fetch)
        0x0F, 0xB6, 0x47, 0xFF,       // movzx eax, byte [edi-1]   <- fetch
        0x30, 0xD8,                   // xor al, bl            (key mix)
        0x14, 0x10,                   // adc al, 0x10          (CF=0 after xor)
        0x30, 0xC8,                   // xor al, cl            (= xor 0x55)
        0xC0, 0xC8, 0x03,             // ror al, 3
        0x30, 0xC3,                   // xor bl, al            (key update)
        0xB8, 0x78, 0x56, 0x34, 0x12, // mov eax, 0x12345678
        0xC3,                         // ret
    ]
}

fn truth(raw: u8, key: u8) -> u8 {
    (((raw ^ key).wrapping_add(0x10)) ^ 0x55).rotate_right(3)
}

/// ELF32 EXEC, one R+X PT_LOAD at VA (zero-fill tail), no section table.
fn fixture_elf32(code: &[u8]) -> PEBinary {
    let mut code = code.to_vec();
    code.resize(LEN as usize, 0xCC);
    let mut img = vec![0u8; 0x100];
    img[0..4].copy_from_slice(b"\x7fELF");
    img[4] = 1; // ELFCLASS32
    img[5] = 1; // little endian
    img[6] = 1; // EV_CURRENT
    img[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    img[18..20].copy_from_slice(&3u16.to_le_bytes()); // EM_386
    img[20..24].copy_from_slice(&1u32.to_le_bytes());
    img[24..28].copy_from_slice(&(VA as u32).to_le_bytes()); // e_entry
    img[28..32].copy_from_slice(&52u32.to_le_bytes()); // e_phoff
    img[40..42].copy_from_slice(&52u16.to_le_bytes()); // e_ehsize
    img[42..44].copy_from_slice(&32u16.to_le_bytes()); // e_phentsize
    img[44..46].copy_from_slice(&1u16.to_le_bytes()); // e_phnum
    img[46..48].copy_from_slice(&40u16.to_le_bytes()); // e_shentsize
    let ph = 52;
    let w = |img: &mut Vec<u8>, o: usize, v: u32| img[ph + o..ph + o + 4].copy_from_slice(&v.to_le_bytes());
    w(&mut img, 0, 1); // PT_LOAD
    w(&mut img, 4, 0x100); // p_offset
    w(&mut img, 8, VA as u32); // p_vaddr
    w(&mut img, 12, VA as u32); // p_paddr
    w(&mut img, 16, code.len() as u32); // p_filesz
    w(&mut img, 20, code.len() as u32 + 0x100); // p_memsz (zero tail)
    w(&mut img, 24, 5); // PF_R | PF_X
    w(&mut img, 28, 0x1000); // p_align
    img.extend_from_slice(&code);
    PEBinary { path: "<elf32>".into(), data: img, fmt: BinFmt::Elf }
}

#[test]
fn sectionless_elf32_maps_via_pt_load() {
    let bin = fixture_elf32(&handler());
    assert_eq!(bin.decode_bits(), 32);
    assert_eq!(bin.image_base().unwrap(), VA);
    let map = bin.section_map().unwrap();
    assert_eq!(map, vec![(VA, VA + LEN, 0x100)]);
    assert_eq!(bin.read_bytes(FETCH, 4).unwrap(), vec![0x0F, 0xB6, 0x47, 0xFF]);
    // Zero-fill tail (memsz > filesz) has no file backing.
    assert!(bin.read_bytes(VA + LEN + 8, 1).is_err());
    let regions = bin.elf_exec_regions().unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].0, "load0");
    assert_eq!(regions[0].1, VA);
}

#[test]
fn scan_32bit_fetch_with_displacement() {
    let code = handler();
    // Historic disp-free rule rejects `[edi-1]`; |disp|<=8 accepts it.
    assert!(scan_fetch_sites_strict_bits(&code, VA, 32, 0).is_empty());
    let sites = scan_fetch_sites_strict_bits(&code, VA, 32, 8);
    assert_eq!(sites.len(), 1);
    assert_eq!(sites[0].va, FETCH);
    assert_eq!(sites[0].base, Register::EDI);
    assert_eq!(sites[0].dst, Register::EAX);
}

#[test]
fn static_miner_decodes_32bit_but_misses_runtime_operands() {
    let bin = fixture_elf32(&handler());
    let site = FetchSite { va: FETCH, base: Register::EDI, dst: Register::EAX, len: 4 };
    let m = mine_cryptor(&site, &bin).unwrap();
    assert_eq!(m.key_reg, "bl");
    // adc (carry) and `xor al,cl` (register operand) are not static ops:
    // the static chain is incomplete here — the reason emu_miner exists.
    assert!((0..=255u8).any(|r| m.decode(r, 0) != truth(r, 0)));
}

#[test]
fn emu_miner_recovers_concrete_chain_from_handler_entry() {
    let bin = fixture_elf32(&handler());
    let image = image_of(&bin).unwrap();
    let m = mine_cryptor_emu(&image, VA, 32).unwrap();
    assert_eq!(m.fetch_va, FETCH);
    assert_eq!(m.vip_reg, "edi");
    assert_eq!(m.key_reg, "bl");
    assert_eq!(m.key_update_va, VA + 18);
    assert!(m.unmodeled.is_empty(), "{:?}", m.unmodeled);
    assert!(m.entry_independent);
    assert!(m.verified);
    // adc al,0x10 with CF=0 -> Add 0x10; xor al,cl with CL=0x55 -> Xor 0x55.
    let got: Vec<(CryptOp, u64)> = m.cryptor.cmds.iter().map(|c| (c.op, c.value)).collect();
    assert_eq!(got, vec![(CryptOp::Add, 0x10), (CryptOp::Xor, 0x55), (CryptOp::Ror, 3)]);
    for raw in 0..=255u8 {
        for key in [0u8, 0x5a, 0xff] {
            assert_eq!(m.cryptor.encrypt((raw ^ key) as u64) as u8, truth(raw, key));
        }
    }
}

#[test]
fn emu_miner_flags_entry_state_dependence_at_fetch_site() {
    // Starting at the fetch skips `mov ecx,0x55`: the chain then uses
    // whatever ECX held, which the perturbed pass exposes.
    let bin = fixture_elf32(&handler());
    let image = image_of(&bin).unwrap();
    let m = mine_cryptor_emu(&image, FETCH, 32).unwrap();
    assert_eq!(m.fetch_va, FETCH);
    assert!(!m.entry_independent);
    assert!(!m.verified);
}
