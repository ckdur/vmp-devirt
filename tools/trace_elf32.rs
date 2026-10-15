//! x86-32 ELF live tracer (Unicorn): runs a dynamically linked i386 ELF
//! from its entry (or any VA) with host-side libc stubs, and records the
//! VM as it executes.
//!
//! usage: trace_elf32 <elf> [start-va]
//!   start-va  default: ELF entry. Starting at `_start`, the
//!             `__libc_start_main` stub runs init(argc,argv,envp) then
//!             main(argc,argv,envp) through a trampoline, like glibc.
//! env:
//!   DATA_DIR     output dir (default ./data)
//!   BOUND        max instructions (default 50000000)
//!   TSC          value returned by rdtsc (default 0). VMP timing-gated
//!                dispatch tables hold equivalent continuations; pinning
//!                the counter picks one deterministically.
//!   ARGV         space-separated argv (default: basename of <elf>)
//!   LOG_READS=lo:hi[,lo:hi]  log data reads in these ranges to reads.jsonl
//!                (eip, handler index, address, size, value)
//!   POKE=va=val[,va=val]  dword writes after loading (state a real
//!                loader/runtime would have set; every poke is printed)
//!   LOG=new|all  new (default): write a dispatch edge (from,handler) or a
//!                fetch (site,vip) only the first time it is seen, so long
//!                loops (integrity hashing) cost no output; counts are in
//!                summary.json. all: every event.
//!   TRACE_DUMP=1 also write trace.bin (every executed VA, u64LE; for
//!                `devirt dispatch`)
//!
//! outputs (DATA_DIR):
//!   handlers.jsonl  one line per indirect `jmp` (VM dispatch): source,
//!                   target, live GPRs/EFLAGS at the target.
//!   fetches.jsonl   decrypted VM operands, live: a register load of 1/2/4
//!                   bytes, then `xor dst,key`, then `xor key,dst` (rolling
//!                   key update) -> site, VIP address, raw, key, decoded.
//!                   No cryptor model involved: values are what ran.
//!   paths.jsonl     executed instruction path of each handler, first
//!                   occurrence only: entry, VAs up to and including the
//!                   next dispatch `jmp` (input of `devirt lift32`).
//!   imports.jsonl   libc stub calls (name, args, return value).
//!   syscalls.jsonl  int 0x80 calls (all return 0 except exit/kill stop).
//!   summary.json    counts + end reason.
//!
//! Harness lies (documented): libc is host stubs (unknown functions return
//! 0), data imports point at zeroed storage, no TLS beyond a zeroed GS
//! page, rdtsc is constant, unmapped data pages are mapped zeroed on use.

use anyhow::{Context, Result};
use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind, Register};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;
use unicorn_engine::unicorn_const::{Arch, HookType, Mode, Prot};
use unicorn_engine::{RegisterX86, Unicorn};
use vmp_devirt::pe_loader::PEBinary;

const STUB: u64 = 0x0a00_0000; // libc stubs, 16 bytes each
const TRAMP: u64 = 0x0a0f_0000; // __libc_start_main trampoline
const OBJ: u64 = 0x0a10_0000; // data-import storage, 0x100 each
const SENTINEL: u64 = 0x0b00_0000; // "returned" / "exit" target
const HEAP: u64 = 0x2000_0000;
const HEAPSZ: u64 = 0x1000_0000;
const STK_TOP: u64 = 0x7f00_0000;
const STKSZ: u64 = 0x20_0000;
const TLS: u64 = 0x0a20_0000;

#[derive(Clone, Copy, Debug)]
enum Info {
    Rdtsc,
    IndJmp,
    /// Register load from memory (mov/movzx/movsx/or r, [m]).
    Load { dst: Register },
    /// `xor r, r'` (same width).
    XorRR { dst: Register, src: Register },
    Other,
}

struct Cand {
    site: u64,
    view: Register,
    size: usize,
    addr: u64,
    raw: u64,
    key: Option<(Register, u64)>,
    born: u64,
}

#[derive(Default)]
struct St {
    n: u64,
    prev: Option<(u64, Info)>,
    last_read: Option<(u64, usize, u64)>,
    /// Current insn is a register load: the read hook records its value.
    want_read: bool,
    seen_edges: std::collections::HashSet<(u64, u64)>,
    seen_fetch: std::collections::HashSet<(u64, u64)>,
    /// Path since the last dispatch (None = not recording: stub call or
    /// handler already recorded).
    path: Option<(u64, Vec<u64>)>,
    seen_paths: std::collections::HashSet<u64>,
    paths_n: u64,
    pending: Vec<Cand>,
    handler_n: u64,
    heap: u64,
    stop: Option<String>,
    fetch_n: u64,
}

fn gpr(r: Register) -> Option<RegisterX86> {
    use Register::*;
    Some(match r.full_register32() {
        EAX => RegisterX86::EAX, ECX => RegisterX86::ECX, EDX => RegisterX86::EDX,
        EBX => RegisterX86::EBX, ESP => RegisterX86::ESP, EBP => RegisterX86::EBP,
        ESI => RegisterX86::ESI, EDI => RegisterX86::EDI,
        _ => return Option::None,
    })
}

fn rd(emu: &Unicorn<()>, r: Register) -> u64 {
    let Some(g) = gpr(r) else { return 0 };
    let v = emu.reg_read(g).unwrap_or(0);
    let hi = matches!(r, Register::AH | Register::CH | Register::DH | Register::BH);
    match r.size() {
        1 if hi => (v >> 8) & 0xff,
        1 => v & 0xff,
        2 => v & 0xffff,
        _ => v & 0xffff_ffff,
    }
}

/// Sub-register of `r` holding `size` bytes (ECX,1 -> CL). None when the
/// view does not exist in 32-bit mode (ESI/EDI/EBP/ESP have no byte view).
fn view(r: Register, size: usize) -> Option<Register> {
    use Register::*;
    let f = r.full_register32();
    Some(match (f, size) {
        (_, 4) => f,
        (EAX, 2) => AX, (ECX, 2) => CX, (EDX, 2) => DX, (EBX, 2) => BX,
        (ESI, 2) => SI, (EDI, 2) => DI, (EBP, 2) => BP, (ESP, 2) => SP,
        (EAX, 1) => AL, (ECX, 1) => CL, (EDX, 1) => DL, (EBX, 1) => BL,
        _ => return Option::None,
    })
}

fn classify(emu: &Unicorn<()>, va: u64) -> Info {
    let Ok(b) = emu.mem_read_as_vec(va, 16) else { return Info::Other };
    if b[0] == 0x0f && b[1] == 0x31 {
        return Info::Rdtsc;
    }
    let mut d = Decoder::with_ip(32, &b, va, DecoderOptions::NONE);
    let ins = d.decode();
    match ins.mnemonic() {
        Mnemonic::Jmp if matches!(ins.op0_kind(), OpKind::Register | OpKind::Memory) => Info::IndJmp,
        Mnemonic::Mov | Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Or
            if ins.op_count() == 2 && ins.op0_kind() == OpKind::Register
                && ins.op1_kind() == OpKind::Memory
                && matches!(ins.memory_size().size(), 1 | 2 | 4)
                && ins.memory_segment() != Register::GS =>
        {
            Info::Load { dst: ins.op0_register() }
        }
        Mnemonic::Xor if ins.op_count() == 2 && ins.op0_kind() == OpKind::Register
            && ins.op1_kind() == OpKind::Register
            && ins.op0_register().size() == ins.op1_register().size()
            && ins.op0_register() != ins.op1_register() =>
        {
            Info::XorRR { dst: ins.op0_register(), src: ins.op1_register() }
        }
        _ => Info::Other,
    }
}

fn regs_json(emu: &Unicorn<()>) -> String {
    let g = |r| emu.reg_read(r).unwrap_or(0);
    format!("\"eax\":{},\"ecx\":{},\"edx\":{},\"ebx\":{},\"esp\":{},\"ebp\":{},\"esi\":{},\"edi\":{},\"eflags\":{}",
        g(RegisterX86::EAX), g(RegisterX86::ECX), g(RegisterX86::EDX), g(RegisterX86::EBX),
        g(RegisterX86::ESP), g(RegisterX86::EBP), g(RegisterX86::ESI), g(RegisterX86::EDI),
        g(RegisterX86::EFLAGS))
}

fn u32_at(emu: &Unicorn<()>, va: u64) -> u64 {
    emu.mem_read_as_vec(va, 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64).unwrap_or(0)
}

fn cstr(emu: &Unicorn<()>, va: u64) -> Option<String> {
    let mut out = Vec::new();
    for i in 0..256 {
        let b = emu.mem_read_as_vec(va + i, 1).ok()?[0];
        if b == 0 {
            break;
        }
        out.push(b);
    }
    Some(String::from_utf8_lossy(&out).into_owned())
}

/// Bump allocator in the heap region (16-byte aligned, never reused).
fn alloc(st: &mut St, n: u64) -> u64 {
    let a = st.heap;
    st.heap += (n.max(16) + 15) & !15;
    a
}

/// Host-side libc semantics. Returns Some(eax) to return to the caller,
/// None when the stub redirected control itself (or stopped).
fn libc(emu: &mut Unicorn<()>, st: &mut St, name: &str, a: &[u64; 6], esp: u64) -> Option<u64> {
    let w = |emu: &mut Unicorn<()>, va: u64, b: &[u8]| { let _ = emu.mem_write(va, b); };
    Some(match name {
        "malloc" | "_Znwj" | "_Znaj" | "valloc" => alloc(st, a[0]),
        "calloc" => alloc(st, a[0].wrapping_mul(a[1])),
        "realloc" => {
            let r = alloc(st, a[1]);
            if a[0] != 0 {
                if let Ok(b) = emu.mem_read_as_vec(a[0], (a[1] as usize).min(0x10000)) {
                    w(emu, r, &b);
                }
            }
            r
        }
        "posix_memalign" => { let r = alloc(st, a[2] + a[1]); let r = (r + a[1] - 1) & !(a[1].max(1) - 1); w(emu, a[0], &(r as u32).to_le_bytes()); 0 }
        "free" | "_ZdlPv" | "_ZdaPv" => 0,
        "memcpy" | "memmove" | "__memcpy_chk" | "__memmove_chk" => {
            if a[2] < 0x100_0000 {
                if let Ok(b) = emu.mem_read_as_vec(a[1], a[2] as usize) { w(emu, a[0], &b); }
            }
            a[0]
        }
        "memset" | "__memset_chk" => { if a[2] < 0x100_0000 { w(emu, a[0], &vec![a[1] as u8; a[2] as usize]); } a[0] }
        "memcmp" | "strncmp" | "strcmp" => {
            let n = if name == "strcmp" { 4096 } else { a[2] as usize };
            let mut r = 0i64;
            for i in 0..n as u64 {
                let x = emu.mem_read_as_vec(a[0] + i, 1).map(|b| b[0]).unwrap_or(0);
                let y = emu.mem_read_as_vec(a[1] + i, 1).map(|b| b[0]).unwrap_or(0);
                if x != y { r = x as i64 - y as i64; break; }
                if x == 0 && name != "memcmp" { break; }
            }
            r as u64 & 0xffff_ffff
        }
        "strlen" => cstr(emu, a[0]).map(|s| s.len() as u64).unwrap_or(0),
        "strcpy" | "__strcpy_chk" => { let s = cstr(emu, a[1]).unwrap_or_default(); let mut b = s.into_bytes(); b.push(0); w(emu, a[0], &b); a[0] }
        "strdup" => { let s = cstr(emu, a[0]).unwrap_or_default(); let r = alloc(st, s.len() as u64 + 1); let mut b = s.into_bytes(); b.push(0); w(emu, r, &b); r }
        "sysconf" | "getpagesize" => 4096,
        "getpid" | "getppid" | "getuid" | "geteuid" | "getgid" => 1,
        "time" => { if a[0] != 0 { w(emu, a[0], &0x5f00_0000u32.to_le_bytes()); } 0x5f00_0000 }
        "gettimeofday" => { w(emu, a[0], &[0u8; 8]); 0 }
        "clock_gettime" => { w(emu, a[1], &[0u8; 8]); 0 }
        "__errno_location" => OBJ + 0xff00,
        "__libc_start_main" => {
            // (main, argc, argv, init, fini, rtld_fini, stack_end)
            let (main, argc, argv, init) = (a[0], a[1], a[2], a[3]);
            let envp = argv + 4 * (argc + 1);
            let mut c: Vec<u8> = Vec::new();
            let push_args = |c: &mut Vec<u8>| {
                for v in [envp, argv, argc] { c.push(0x68); c.extend_from_slice(&(v as u32).to_le_bytes()); }
            };
            if init != 0 {
                push_args(&mut c);
                c.push(0xB8); c.extend_from_slice(&(init as u32).to_le_bytes()); c.extend_from_slice(&[0xFF, 0xD0]);
                c.extend_from_slice(&[0x83, 0xC4, 0x0C]);
            }
            push_args(&mut c);
            c.push(0xB8); c.extend_from_slice(&(main as u32).to_le_bytes()); c.extend_from_slice(&[0xFF, 0xD0]);
            c.extend_from_slice(&[0x83, 0xC4, 0x0C, 0x50]); // add esp,12; push eax
            c.push(0xB8); c.extend_from_slice(&(SENTINEL as u32).to_le_bytes()); c.extend_from_slice(&[0xFF, 0xE0]);
            w(emu, TRAMP, &c);
            let _ = emu.reg_write(RegisterX86::ESP, esp & !0xf);
            let _ = emu.reg_write(RegisterX86::EIP, TRAMP);
            eprintln!("__libc_start_main: init={:#x} main={:#x}", init, main);
            return None;
        }
        "exit" | "_exit" | "abort" | "__stack_chk_fail" | "__assert_fail" => {
            st.stop = Some(format!("{}({:#x})", name, a[0]));
            let _ = emu.emu_stop();
            return None;
        }
        _ => 0,
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: trace_elf32 <elf> [start-va]   (env: DATA_DIR BOUND TSC ARGV TRACE_DUMP)");
        std::process::exit(2);
    }
    let bin = PEBinary::load(&args[1]).with_context(|| format!("load {}", args[1]))?;
    anyhow::ensure!(bin.decode_bits() == 32, "not a 32-bit ELF");
    let elf = bin.parse_elf()?;
    let start = match args.get(2) {
        Some(s) => u64::from_str_radix(s.trim_start_matches("0x"), 16)?,
        None => elf.entry,
    };
    let env_u64 = |k: &str, d: u64| std::env::var(k).ok()
        .and_then(|v| if let Some(h) = v.strip_prefix("0x") { u64::from_str_radix(h, 16).ok() } else { v.parse().ok() })
        .unwrap_or(d);
    let bound = env_u64("BOUND", 50_000_000);
    let tsc = env_u64("TSC", 0);
    let dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".into());
    std::fs::create_dir_all(&dir)?;

    let mut emu = Unicorn::new(Arch::X86, Mode::MODE_32).map_err(|e| anyhow::anyhow!("uc: {:?}", e))?;
    // Image: page-union of every mapped section/segment, then bytes.
    let mut spans: Vec<(u64, u64)> = bin.map_sections()?.iter()
        .map(|(_, va, _, _, virt)| (va & !0xfff, (va + *virt as u64 + 0xfff) & !0xfff)).collect();
    spans.sort();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (lo, hi) in spans {
        match merged.last_mut() { Some(m) if lo <= m.1 => m.1 = m.1.max(hi), _ => merged.push((lo, hi)) }
    }
    for (lo, hi) in &merged {
        emu.mem_map(*lo, hi - lo, Prot::ALL).map_err(|e| anyhow::anyhow!("map {:#x}: {:?}", lo, e))?;
    }
    for (_, va, off, raw, _) in bin.map_sections()? {
        if raw > 0 { let _ = emu.mem_write(va, &bin.data[off..(off + raw).min(bin.data.len())]); }
    }
    for (va, sz) in [(STUB, 0x10_0000u64), (OBJ, 0x10_0000), (TLS, 0x1000), (HEAP, HEAPSZ), (STK_TOP - STKSZ, STKSZ)] {
        emu.mem_map(va, sz, Prot::ALL).map_err(|e| anyhow::anyhow!("map {:#x}: {:?}", va, e))?;
    }
    // Minimal TLS: GS base -> zeroed page (stack-protector canary gs:[0x14] = 0).
    let _ = emu.reg_write(RegisterX86::GS_BASE, TLS);

    // Imports: JUMP_SLOT / GLOB_DAT / R_386_32 -> function stubs or zeroed objects.
    let mut stubs: HashMap<u64, String> = HashMap::new();
    let mut nobj = 0u64;
    for r in elf.pltrelocs.iter().chain(elf.dynrels.iter()) {
        let Some(sym) = elf.dynsyms.get(r.r_sym) else { continue };
        let name = elf.dynstrtab.get_at(sym.st_name).unwrap_or("").to_string();
        if name.is_empty() || sym.st_value != 0 { continue; }
        let target = if r.r_type == 7 || sym.st_type() == goblin::elf::sym::STT_FUNC
            || (sym.st_type() == goblin::elf::sym::STT_NOTYPE && !name.starts_with("_ZTV"))
        {
            if name == "__gmon_start__" || name == "_Jv_RegisterClasses" { 0 } else {
                let a = STUB + stubs.len() as u64 * 16;
                let _ = emu.mem_write(a, &[0xC3]);
                stubs.insert(a, name.clone());
                a
            }
        } else {
            nobj += 1;
            OBJ + nobj * 0x100
        };
        let _ = emu.mem_write(r.r_offset, &(target as u32).to_le_bytes());
    }
    eprintln!("imports: {} function stubs, {} data objects", stubs.len(), nobj);

    for kv in std::env::var("POKE").unwrap_or_default().split(',').filter(|x| !x.is_empty()) {
        let (a, v) = kv.split_once('=').context("POKE wants va=val")?;
        let p = |x: &str| u64::from_str_radix(x.trim().trim_start_matches("0x"), 16);
        let (a, v) = (p(a)?, p(v)?);
        emu.mem_write(a, &(v as u32).to_le_bytes()).map_err(|e| anyhow::anyhow!("poke {:#x}: {:?}", a, e))?;
        eprintln!("poke [{:#x}] = {:#x}", a, v);
    }
    // Process stack: argc, argv[], NULL, envp NULL, strings.
    let argv_s: Vec<String> = std::env::var("ARGV").map(|v| v.split_whitespace().map(String::from).collect())
        .unwrap_or_else(|_| vec![std::path::Path::new(&args[1]).file_name().unwrap().to_string_lossy().into()]);
    let mut strp = STK_TOP - 0x1000;
    let mut ptrs = Vec::new();
    for s in &argv_s {
        let mut b = s.clone().into_bytes(); b.push(0);
        let _ = emu.mem_write(strp, &b);
        ptrs.push(strp);
        strp += b.len() as u64 + 1;
    }
    let esp = STK_TOP - 0x2000;
    let mut frame: Vec<u8> = Vec::new();
    if start != elf.entry {
        frame.extend_from_slice(&(SENTINEL as u32).to_le_bytes()); // return address
    }
    frame.extend_from_slice(&(argv_s.len() as u32).to_le_bytes());
    let argv_va = esp + frame.len() as u64 + 4 * if start != elf.entry { 2 } else { 0 };
    if start != elf.entry {
        // called as f(argc, argv, envp)
        frame.extend_from_slice(&(argv_va as u32).to_le_bytes());
        frame.extend_from_slice(&((argv_va + 4 * (ptrs.len() as u64 + 1)) as u32).to_le_bytes());
    }
    for p in &ptrs { frame.extend_from_slice(&(*p as u32).to_le_bytes()); }
    frame.extend_from_slice(&[0u8; 12]); // argv NULL, envp NULL, auxv NULL
    let _ = emu.mem_write(esp, &frame);
    emu.reg_write(RegisterX86::ESP, esp).ok();
    emu.reg_write(RegisterX86::EFLAGS, 0x202).ok();

    let out = |n: &str| -> Result<std::io::BufWriter<std::fs::File>> {
        Ok(std::io::BufWriter::new(std::fs::File::create(format!("{}/{}", dir, n))?))
    };
    let f_handlers = Rc::new(RefCell::new(out("handlers.jsonl")?));
    let f_fetches = Rc::new(RefCell::new(out("fetches.jsonl")?));
    let f_imports = Rc::new(RefCell::new(out("imports.jsonl")?));
    let f_paths = Rc::new(RefCell::new(out("paths.jsonl")?));
    let f_sys = Rc::new(RefCell::new(out("syscalls.jsonl")?));
    let f_trace = if std::env::var("TRACE_DUMP").is_ok() { Some(Rc::new(RefCell::new(out("trace.bin")?))) } else { None };

    let st = Rc::new(RefCell::new(St { heap: HEAP, ..Default::default() }));
    // Decode cache: flat over the image span, map elsewhere (stubs, heap).
    let (img_lo, img_hi) = (merged.first().map(|m| m.0).unwrap_or(0), merged.last().map(|m| m.1).unwrap_or(0));
    let flat: Rc<RefCell<Vec<Option<Info>>>> = Rc::new(RefCell::new(vec![None; (img_hi - img_lo) as usize]));
    let cache: Rc<RefCell<HashMap<u64, Info>>> = Rc::new(RefCell::new(HashMap::new()));
    let log_all = std::env::var("LOG").map(|v| v == "all").unwrap_or(false);

    // Lazily map data pages (logged); instruction fetches stay fatal.
    let faults = Rc::new(RefCell::new(0u64));
    let fl = faults.clone();
    emu.add_mem_hook(HookType::MEM_READ_UNMAPPED | HookType::MEM_WRITE_UNMAPPED, 0, u64::MAX, move |emu, _t, addr, size, _v| {
        let lo = addr & !0xfff;
        let hi = (addr + size as u64 + 0xfff) & !0xfff;
        let mut c = fl.borrow_mut();
        *c += 1;
        if *c <= 20 { eprintln!("  lazy-map {:#x} (eip {:#x})", addr, emu.reg_read(RegisterX86::EIP).unwrap_or(0)); }
        (lo..hi).step_by(0x1000).all(|p| emu.mem_read_as_vec(p, 1).is_ok() || emu.mem_map(p, 0x1000, Prot::ALL).is_ok())
    }).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let log_ranges: Vec<(u64, u64)> = std::env::var("LOG_READS").unwrap_or_default().split(',')
        .filter_map(|r| {
            let (a, b) = r.split_once(':')?;
            Some((u64::from_str_radix(a.trim_start_matches("0x"), 16).ok()?, u64::from_str_radix(b.trim_start_matches("0x"), 16).ok()?))
        }).collect();
    let f_reads = if log_ranges.is_empty() { None } else { Some(Rc::new(RefCell::new(out("reads.jsonl")?))) };
    let fr = f_reads.clone();
    let s_r = st.clone();
    emu.add_mem_hook(HookType::MEM_READ, 0, u64::MAX, move |emu, _t, addr, size, _v| {
        let mut s = s_r.borrow_mut();
        if !s.want_read && fr.is_none() {
            return true;
        }
        let mut x = [0u8; 8];
        let _ = emu.mem_read(addr, &mut x[..size.min(8)]);
        let v = u64::from_le_bytes(x);
        if s.want_read {
            s.last_read = Some((addr, size, v));
        }
        if let Some(f) = &fr {
            if log_ranges.iter().any(|(lo, hi)| addr >= *lo && addr < *hi) {
                let _ = writeln!(f.borrow_mut(), "{{\"insn\":{},\"handler_n\":{},\"eip\":{},\"addr\":{},\"size\":{},\"val\":{}}}",
                    s.n, s.handler_n, emu.reg_read(RegisterX86::EIP).unwrap_or(0), addr, size, v);
            }
        }
        true
    }).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let s_i = st.clone();
    let fs = f_sys.clone();
    emu.add_intr_hook(move |emu, intno| {
        let g = |r| emu.reg_read(r).unwrap_or(0);
        let (nr, b, c, d) = (g(RegisterX86::EAX), g(RegisterX86::EBX), g(RegisterX86::ECX), g(RegisterX86::EDX));
        let _ = writeln!(fs.borrow_mut(), "{{\"int\":{},\"nr\":{},\"ebx\":{},\"ecx\":{},\"edx\":{},\"eip\":{}}}", intno, nr, b, c, d, g(RegisterX86::EIP));
        let mut s = s_i.borrow_mut();
        // exit, exit_group, kill: end of run (kill(getpid,9) = anti-debug).
        if intno == 0x80 && matches!(nr, 1 | 252 | 37) {
            s.stop = Some(format!("syscall {} ({:#x},{:#x})", nr, b, c));
            let _ = emu.emu_stop();
        }
        let _ = emu.reg_write(RegisterX86::EAX, if nr == 20 { 1 } else { 0 });
    }).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let s_c = st.clone();
    let (fh, ff, fi, ft, fp) = (f_handlers.clone(), f_fetches.clone(), f_imports.clone(), f_trace.clone(), f_paths.clone());
    let stubs_c = stubs.clone();
    let cache_c = cache.clone();
    let flat_c = flat.clone();
    emu.add_code_hook(0, u64::MAX, move |emu, addr, _size| {
        let mut s = s_c.borrow_mut();
        s.n += 1;
        if s.n > bound {
            s.stop = Some("bound".into());
            let _ = emu.emu_stop();
            return;
        }
        if let Some(t) = &ft { let _ = t.borrow_mut().write_all(&addr.to_le_bytes()); }
        if addr == SENTINEL {
            s.stop = Some(format!("returned eax={:#x}", emu.reg_read(RegisterX86::EAX).unwrap_or(0)));
            let _ = emu.emu_stop();
            return;
        }
        if let Some(name) = stubs_c.get(&addr) {
            let esp = emu.reg_read(RegisterX86::ESP).unwrap_or(0);
            let ret = u32_at(emu, esp);
            let a: [u64; 6] = std::array::from_fn(|i| u32_at(emu, esp + 4 + 4 * i as u64));
            let strs: Vec<String> = a[..2].iter().map(|p| if *p > 0x1000 { cstr(emu, *p).filter(|x| x.len() > 1 && x.is_ascii()).unwrap_or_default() } else { String::new() }).collect();
            let r = libc(emu, &mut s, name, &a, esp);
            let _ = writeln!(fi.borrow_mut(), "{{\"n\":{},\"name\":\"{}\",\"args\":[{}],\"ret_to\":{},\"ret\":{},\"s0\":{:?},\"s1\":{:?}}}",
                s.n, name, a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), ret,
                r.map(|x| x.to_string()).unwrap_or("null".into()), strs[0], strs[1]);
            if let Some(r) = r {
                let _ = emu.reg_write(RegisterX86::EAX, r);
                let _ = emu.reg_write(RegisterX86::ESP, esp + 4);
                let _ = emu.reg_write(RegisterX86::EIP, ret);
            }
            s.prev = None;
            s.path = None; // native call: not a VM handler path
            return;
        }
        let info = if addr >= img_lo && addr < img_hi {
            let mut f = flat_c.borrow_mut();
            let slot = &mut f[(addr - img_lo) as usize];
            *slot.get_or_insert_with(|| classify(emu, addr))
        } else {
            *cache_c.borrow_mut().entry(addr).or_insert_with(|| classify(emu, addr))
        };
        if s.n % 1_000_000_000 == 0 {
            eprintln!("  .. {} insns, {} dispatches ({} edges), {} fetches ({} unique), eip {:#x}",
                s.n, s.handler_n, s.seen_edges.len(), s.fetch_n, s.seen_fetch.len(), addr);
        }
        // Dispatch target: record live context at handler entry.
        if let Some((from, Info::IndJmp)) = s.prev {
            if let Some((entry, p)) = s.path.take() {
                if s.seen_paths.insert(entry) && p.len() > 1 {
                    s.paths_n += 1;
                    let _ = writeln!(fp.borrow_mut(), "{{\"handler\":{},\"insns\":[{}]}}",
                        entry, p.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
                }
            }
            if !s.seen_paths.contains(&addr) {
                s.path = Some((addr, Vec::with_capacity(256)));
            }
            s.handler_n += 1;
            if s.seen_edges.insert((from, addr)) || log_all {
                let _ = writeln!(fh.borrow_mut(), "{{\"n\":{},\"insn\":{},\"from\":{},\"handler\":{},{}}}",
                    s.handler_n, s.n, from, addr, regs_json(emu));
            }
        }
        // Resolve the previous insn's load into a fetch candidate.
        if let (Some((pva, Info::Load { dst })), Some((raddr, rsize, rval))) = (s.prev, s.last_read.take()) {
            if let Some(v) = view(dst, rsize) {
                let full = dst.full_register32();
                s.pending.retain(|c| c.view.full_register32() != full);
                let born = s.n;
                s.pending.push(Cand { site: pva, view: v, size: rsize, addr: raddr, raw: rval, key: None, born });
                if s.pending.len() > 6 { s.pending.remove(0); }
            }
        }
        s.last_read = None;
        s.want_read = matches!(info, Info::Load { .. });
        if let Some((_, p)) = s.path.as_mut() {
            if p.len() < 20000 { p.push(addr); } else { s.path = None; }
        }
        match info {
            Info::Rdtsc => {
                let _ = emu.reg_write(RegisterX86::EAX, tsc & 0xffff_ffff);
                let _ = emu.reg_write(RegisterX86::EDX, tsc >> 32);
                let _ = emu.reg_write(RegisterX86::EIP, addr + 2);
                s.prev = None;
                return;
            }
            Info::XorRR { dst, src } => {
                let now = s.n;
                let mut done: Option<usize> = None;
                for i in (0..s.pending.len()).rev() {
                    let c = &mut s.pending[i];
                    if now - c.born > 400 { continue; }
                    match c.key {
                        None if dst == c.view && src.size() == c.size => {
                            c.key = Some((src, rd(emu, src)));
                            break;
                        }
                        Some((k, _)) if dst == k && src == c.view => { done = Some(i); break; }
                        _ => {}
                    }
                }
                if let Some(i) = done {
                    let c = s.pending.remove(i);
                    let dec = rd(emu, c.view);
                    let (k, kv) = c.key.unwrap();
                    s.fetch_n += 1;
                    if s.seen_fetch.insert((c.site, c.addr)) || log_all {
                    let _ = writeln!(ff.borrow_mut(), "{{\"n\":{},\"handler_n\":{},\"site\":{},\"size\":{},\"vip\":{},\"raw\":{},\"key_reg\":\"{:?}\",\"key\":{},\"dec\":{},\"update\":{}}}",
                        s.fetch_n, s.handler_n, c.site, c.size, c.addr, c.raw, k, kv, dec, addr);
                    }
                }
            }
            _ => {}
        }
        let now = s.n;
        s.pending.retain(|c| now - c.born <= 400);
        s.prev = Some((addr, info));
    }).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    eprintln!("start {:#x} bound {} tsc {:#x}", start, bound, tsc);
    let t0 = std::time::Instant::now();
    let res = emu.emu_start(start, u64::MAX, 0, 0);
    let dt = t0.elapsed().as_secs_f64();
    for f in [&f_handlers, &f_fetches, &f_imports, &f_sys, &f_paths] { f.borrow_mut().flush()?; }
    if let Some(f) = &f_reads { f.borrow_mut().flush()?; }
    if let Some(t) = &f_trace { t.borrow_mut().flush()?; }
    let s = st.borrow();
    let eip = emu.reg_read(RegisterX86::EIP).unwrap_or(0);
    let reason = s.stop.clone().unwrap_or_else(|| match &res { Ok(_) => "ok".into(), Err(e) => format!("{:?} at {:#x}", e, eip) });
    let summary = format!("{{\"start\":{},\"insns\":{},\"handlers\":{},\"handler_edges\":{},\"fetches\":{},\"fetch_unique\":{},\"paths\":{},\"lazy_faults\":{},\"end_eip\":{},\"reason\":{:?},\"seconds\":{:.2}}}\n",
        start, s.n, s.handler_n, s.seen_edges.len(), s.fetch_n, s.seen_fetch.len(), s.paths_n, *faults.borrow(), eip, reason, dt);
    std::fs::write(format!("{}/summary.json", dir), &summary)?;
    print!("{}", summary);
    Ok(())
}
