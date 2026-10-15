//! Emulation-backed cryptor miner (stage 1 for mutated builds).
//!
//! The static miner (`cryptor_miner`) pattern-matches ALU ops on the
//! fetch byte. Mutated builds (seen on VMP x86-32 ELF) interleave ops
//! whose operands only exist at run time: `xor cl,al` with a junk-computed
//! `al`, `sbb cl,imm` with a known CF, chains spread over several jmps.
//! Measured on a mutated VMP 3.x x86-32 ELF: static chains match the
//! emulated truth table at 19/219 executed fetch sites; this miner at
//! 216/224 handler entries verified, 0 wrong (docs/coverage.md).
//!
//! Here the code runs in Unicorn once per input byte (0..=255). Every
//! executed op on the fetch byte is recorded with its *concrete* operand
//! (register value, carry-in) and mapped to a `CryptOp`. A shadow model
//! checks after each insn that the chain so far predicts the byte; any
//! write it can't model is reported in `unmodeled` (auditable, not
//! silent). The chain is accepted when all runs agree on the op list
//! and it reproduces every run's output.
//!
//! Setup: every GPR but the stack pointer points into its own buffer
//! filled with the input byte, so whichever register is the VIP (roles
//! are randomized per VM) its fetch reads the input. The live key byte is
//! read at the key mix (it may already be rolled by earlier dword
//! fetches). Fetch = first 1-byte read from a buffer. Key = r8 of the first
//! `xor dst8,r8`; the run stops at the rolling-key update `xor key8,dst8`.
//! Start at the handler entry when ops use registers set before the
//! fetch (junk constants); starting at the fetch itself loses them.

use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};
use std::cell::RefCell;
use std::rc::Rc;
use unicorn_engine::unicorn_const::{Arch, HookType, Mode, Prot};
use unicorn_engine::{RegisterX86, Unicorn};

use crate::backend::value_cryptor::{CryptOp, CryptSize, ValueCryptor};

const BUF: u64 = 0x6000_0000;
const BUFSZ: u64 = 0x2_0000;
/// Perturbed-pass buffers: scattered bases so pointer *high* bits differ
/// too (junk constants like `shr edx,13` read them).
const BUF_ALT: u64 = 0x3000_0000;
const ALT_STRIDE: u64 = 0x0137_0000;
const MAX_GPRS: u64 = 16;

/// Buffer base for GPR `i` in pass `variant`.
fn buf_base(i: u64, variant: u64) -> u64 {
    if variant == 0 { BUF + i * BUFSZ } else { BUF_ALT + i * ALT_STRIDE + 0x5_a000 }
}
const STK: u64 = 0x7f00_0000;
const STKSZ: u64 = 0x10_0000;
const MAX_INSNS: usize = 4000;

/// Emulation-mined decryptor for one entry.
#[derive(Debug, Clone)]
pub struct EmuMined {
    /// Where emulation started (handler entry or fetch site).
    pub entry_va: u64,
    /// The fetch insn (first 1-byte bytecode read).
    pub fetch_va: u64,
    /// Register holding the VIP at the fetch (e.g. "edi").
    pub vip_reg: String,
    /// Rolling-key register (e.g. "bl").
    pub key_reg: String,
    /// The rolling-key update insn (`xor key8,dst8`), end of chain.
    pub key_update_va: u64,
    /// Chain applied after the key mix: opcode = cryptor.encrypt(raw ^ key).
    pub cryptor: ValueCryptor,
    /// Insns that changed the fetch byte outside the model ("va: text").
    pub unmodeled: Vec<String>,
    /// Emulated outputs for raw 0..=255 (key as found in the run).
    pub table: Vec<u8>,
    /// A second pass from a perturbed entry state (other pointer values,
    /// stack fill, status flags) yields the same decode function. False means the chain
    /// uses state set before `entry` (start at the handler entry instead).
    pub entry_independent: bool,
    /// Chain reproduces all 256 outputs, nothing was unmodeled, and the
    /// result is entry-state independent.
    pub verified: bool,
}

struct Run {
    fetch_va: Option<u64>,
    vip: Option<Register>,
    dst8: Option<Register>,
    key8: Option<Register>,
    key_val: u8,
    key_update_va: Option<u64>,
    out: Option<u8>,
    ops: Vec<(CryptOp, u64)>,
    pred: u8,
    check: bool,
    prev: Option<(u64, String)>,
    last_pc: u64,
    unmodeled: Vec<String>,
    n: usize,
}

fn uc_full(r: Register, bits: u32) -> Option<RegisterX86> {
    use Register::*;
    let f = if bits == 32 { r.full_register32() } else { r.full_register() };
    Some(match f {
        EAX => RegisterX86::EAX, ECX => RegisterX86::ECX, EDX => RegisterX86::EDX,
        EBX => RegisterX86::EBX, ESP => RegisterX86::ESP, EBP => RegisterX86::EBP,
        ESI => RegisterX86::ESI, EDI => RegisterX86::EDI,
        RAX => RegisterX86::RAX, RCX => RegisterX86::RCX, RDX => RegisterX86::RDX,
        RBX => RegisterX86::RBX, RSP => RegisterX86::RSP, RBP => RegisterX86::RBP,
        RSI => RegisterX86::RSI, RDI => RegisterX86::RDI,
        R8 => RegisterX86::R8, R9 => RegisterX86::R9, R10 => RegisterX86::R10,
        R11 => RegisterX86::R11, R12 => RegisterX86::R12, R13 => RegisterX86::R13,
        R14 => RegisterX86::R14, R15 => RegisterX86::R15,
        _ => return Option::None,
    })
}

/// Concrete value of any GPR view (8/16/32/64, incl. AH..BH).
fn read_reg(emu: &Unicorn<()>, r: Register, bits: u32) -> Option<u64> {
    let v = emu.reg_read(uc_full(r, bits)?).ok()?;
    let hi = matches!(r, Register::AH | Register::CH | Register::DH | Register::BH);
    Some(match r.size() {
        1 if hi => (v >> 8) & 0xff,
        1 => v & 0xff,
        2 => v & 0xffff,
        4 => v & 0xffff_ffff,
        _ => v,
    })
}

/// Byte view holding the fetched value (`movzx ecx,..` -> CL).
fn low8(r: Register, bits: u32) -> Option<Register> {
    if r.size() == 1 {
        return Some(r);
    }
    let l = crate::frontend::cryptor_miner::low8_of(r);
    // SIL/DIL/BPL/SPL/R*L do not exist in 32-bit mode.
    if bits == 32 && !matches!(l, Register::AL | Register::CL | Register::DL | Register::BL) {
        return None;
    }
    if l.size() == 1 { Some(l) } else { None }
}

fn decode_at(emu: &Unicorn<()>, va: u64, bits: u32) -> Option<Instruction> {
    let b = emu.mem_read_as_vec(va, 16).ok()?;
    let mut d = Decoder::with_ip(bits, &b, va, DecoderOptions::NONE);
    let ins = d.decode();
    if ins.is_invalid() { None } else { Some(ins) }
}

/// Concrete second operand of an ALU op on `dst8` (imm or register).
fn operand(emu: &Unicorn<()>, ins: &Instruction, bits: u32) -> Option<u64> {
    if ins.op_count() < 2 {
        return None;
    }
    match ins.op1_kind() {
        OpKind::Register => read_reg(emu, ins.op1_register(), bits),
        OpKind::Immediate8 | OpKind::Immediate8_2nd => Some(ins.immediate8() as u64),
        _ => None,
    }
}

fn step(op: CryptOp, val: u64, x: u8) -> u8 {
    let mut c = ValueCryptor::new(CryptSize::Byte);
    c.add(op, val);
    c.encrypt(x as u64) as u8
}

/// Map one insn on `dst8` to a chain op with concrete operand.
fn model(emu: &Unicorn<()>, ins: &Instruction, bits: u32) -> Option<(CryptOp, u64)> {
    let cf = emu.reg_read(RegisterX86::EFLAGS).ok()? & 1;
    let v = || operand(emu, ins, bits);
    Some(match ins.mnemonic() {
        Mnemonic::Xor => (CryptOp::Xor, v()? & 0xff),
        Mnemonic::Add => (CryptOp::Add, v()? & 0xff),
        Mnemonic::Adc => (CryptOp::Add, (v()? + cf) & 0xff),
        Mnemonic::Sub => (CryptOp::Sub, v()? & 0xff),
        Mnemonic::Sbb => (CryptOp::Sub, (v()? + cf) & 0xff),
        Mnemonic::And => (CryptOp::And, v()? & 0xff),
        Mnemonic::Inc => (CryptOp::Inc, 0),
        Mnemonic::Dec => (CryptOp::Dec, 0),
        Mnemonic::Not => (CryptOp::Not, 0),
        Mnemonic::Neg => (CryptOp::Neg, 0),
        // x86 masks the count to 5 bits; the 8-bit rotate is then mod 8.
        Mnemonic::Rol => (CryptOp::Rol, v()? & 31),
        Mnemonic::Ror => (CryptOp::Ror, v()? & 31),
        _ => return Option::None,
    })
}

/// Executable image for the emulator: (va, bytes); zero-fill tails ok.
pub type Image = Vec<(u64, Vec<u8>)>;

/// Build the image from a loaded binary (`map_sections`, zero-filled
/// virtual tails).
pub fn image_of(bin: &crate::pe_loader::PEBinary) -> Result<Image> {
    let mut out = Vec::new();
    for (_, va, off, raw, virt) in bin.map_sections()? {
        let mut b = bin.data.get(off..off + raw).unwrap_or(&[]).to_vec();
        b.resize(virt.max(raw), 0);
        out.push((va, b));
    }
    Ok(out)
}

/// Lazily mapped data pages (lo, len); unmapped after every run so each
/// input starts from fresh zero pages.
type LazyPages = Rc<RefCell<Vec<(u64, u64)>>>;

fn make_emu<'a>(image: &Image, bits: u32) -> Result<(Unicorn<'a, ()>, LazyPages)> {
    let mode = if bits == 32 { Mode::MODE_32 } else { Mode::MODE_64 };
    let mut emu = Unicorn::new(Arch::X86, mode).map_err(|e| anyhow::anyhow!("uc: {:?}", e))?;
    // Page-union of all regions (sections can share pages).
    let mut pages: Vec<(u64, u64)> = image.iter()
        .filter(|(_, b)| !b.is_empty())
        .map(|(va, b)| (va & !0xfff, (va + b.len() as u64 + 0xfff) & !0xfff))
        .collect();
    pages.sort();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (lo, hi) in pages {
        match merged.last_mut() {
            Some(m) if lo <= m.1 => m.1 = m.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    for (lo, hi) in merged {
        emu.mem_map(lo, hi - lo, Prot::ALL).map_err(|e| anyhow::anyhow!("map {:#x}: {:?}", lo, e))?;
    }
    for (va, b) in image {
        let _ = emu.mem_write(*va, b);
    }
    emu.mem_map(BUF, BUFSZ * MAX_GPRS, Prot::ALL).map_err(|e| anyhow::anyhow!("map buf: {:?}", e))?;
    for i in 0..MAX_GPRS {
        emu.mem_map(buf_base(i, 1), BUFSZ, Prot::ALL).map_err(|e| anyhow::anyhow!("map alt buf: {:?}", e))?;
    }
    emu.mem_map(STK, STKSZ, Prot::ALL).map_err(|e| anyhow::anyhow!("map stack: {:?}", e))?;
    // Data pages on demand (zeroed): junk/context accesses outside the
    // image. Code fetches from unmapped memory stay fatal.
    let lazy: LazyPages = Rc::new(RefCell::new(Vec::new()));
    let l1 = lazy.clone();
    emu.add_mem_hook(HookType::MEM_READ_UNMAPPED | HookType::MEM_WRITE_UNMAPPED, 0, u64::MAX,
        move |emu, _t, addr, size, _v| {
            let lo = addr & !0xfff;
            let hi = (addr + size as u64 + 0xfff) & !0xfff;
            // Page-wise: a straddling access may hit one mapped page.
            let mut ok = true;
            for pg in (lo..hi).step_by(0x1000) {
                if emu.mem_read_as_vec(pg, 1).is_ok() {
                    continue;
                }
                if emu.mem_map(pg, 0x1000, Prot::ALL).is_ok() {
                    l1.borrow_mut().push((pg, 0x1000));
                } else {
                    ok = false;
                }
            }
            ok
        }).map_err(|e| anyhow::anyhow!("hook: {:?}", e))?;
    Ok((emu, lazy))
}

fn gprs(bits: u32) -> Vec<(Register, RegisterX86)> {
    use Register::*;
    if bits == 32 {
        vec![(EAX, RegisterX86::EAX), (ECX, RegisterX86::ECX), (EDX, RegisterX86::EDX),
             (EBX, RegisterX86::EBX), (EBP, RegisterX86::EBP), (ESI, RegisterX86::ESI),
             (EDI, RegisterX86::EDI)]
    } else {
        vec![(RAX, RegisterX86::RAX), (RCX, RegisterX86::RCX), (RDX, RegisterX86::RDX),
             (RBX, RegisterX86::RBX), (RBP, RegisterX86::RBP), (RSI, RegisterX86::RSI),
             (RDI, RegisterX86::RDI), (R8, RegisterX86::R8), (R9, RegisterX86::R9),
             (R10, RegisterX86::R10), (R11, RegisterX86::R11), (R12, RegisterX86::R12),
             (R13, RegisterX86::R13), (R14, RegisterX86::R14), (R15, RegisterX86::R15)]
    }
}

/// One emulation of `entry` with input byte `raw`. `variant` perturbs
/// the entry state (every pointer byte, stack fill, status flags) so the
/// caller can check the chain does not depend on it.
fn run_one(emu: &mut Unicorn<()>, lazy: &LazyPages, entry: u64, raw: u8, bits: u32, variant: u64) -> Result<Run> {
    let regs = gprs(bits);
    let nbuf = regs.len() as u64;
    let fillbuf = vec![raw; BUFSZ as usize];
    for i in 0..nbuf {
        emu.mem_write(buf_base(i, variant), &fillbuf).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    }
    let fill = (variant as u8).wrapping_mul(0x5a);
    emu.mem_write(STK, &vec![fill; STKSZ as usize]).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    for (i, (_, r)) in regs.iter().enumerate() {
        let p = buf_base(i as u64, variant) + BUFSZ / 2 + variant * (0x357 + 0x2b * i as u64);
        emu.reg_write(*r, p).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    }
    let sp = if bits == 32 { RegisterX86::ESP } else { RegisterX86::RSP };
    emu.reg_write(sp, STK + STKSZ / 2).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    // Perturbed passes also flip CF/PF/AF/ZF/SF/OF (setcc/sbb constants).
    let flags = if variant == 0 { 0x202 } else { 0x202 | 0x8d5 };
    emu.reg_write(RegisterX86::EFLAGS, flags).map_err(|e| anyhow::anyhow!("{:?}", e))?;

    let st = Rc::new(RefCell::new(Run {
        fetch_va: None, vip: None, dst8: None, key8: None, key_val: 0, key_update_va: None,
        out: None, ops: Vec::new(), pred: raw, check: false, prev: None, last_pc: entry,
        unmodeled: Vec::new(), n: 0,
    }));
    // Undo log: every run starts from the same memory (junk writes to the
    // image, stack or lazily mapped pages must not leak between inputs).
    let undo: Rc<RefCell<Vec<(u64, Vec<u8>)>>> = Rc::new(RefCell::new(Vec::new()));
    let u1 = undo.clone();
    let hw = emu.add_mem_hook(HookType::MEM_WRITE, 0, u64::MAX, move |emu, _t, addr, size, _v| {
        if let Ok(old) = emu.mem_read_as_vec(addr, size) {
            u1.borrow_mut().push((addr, old));
        }
        true
    }).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let regs_r = regs.clone();
    let s1 = st.clone();
    let hr = emu.add_mem_hook(HookType::MEM_READ, 0, u64::MAX, move |_emu, _t, addr, size, _v| {
        let mut s = s1.borrow_mut();
        let hit = (0..nbuf).find(|&i| {
            let b = buf_base(i, variant);
            addr >= b && addr < b + BUFSZ
        });
        if let (None, 1, Some(i)) = (s.fetch_va, size, hit) {
            s.fetch_va = Some(s.last_pc);
            s.vip = Some(regs_r[i as usize].0);
        }
        true
    }).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let s2 = st.clone();
    let hc = emu.add_code_hook(0, u64::MAX, move |emu, addr, _size| {
        let mut s = s2.borrow_mut();
        s.n += 1;
        if s.n > MAX_INSNS {
            let _ = emu.emu_stop();
            return;
        }
        // Fetch happened in the previous insn: bind dst8, expect raw.
        if s.fetch_va.is_some() && s.dst8.is_none() {
            let fv = s.fetch_va.unwrap();
            match decode_at(emu, fv, bits).and_then(|i| low8(i.op0_register(), bits)) {
                Some(d) => { s.dst8 = Some(d); s.pred = raw; s.check = true; }
                None => {
                    s.unmodeled.push(format!("{:#x}: fetch dst has no byte view", fv));
                    let _ = emu.emu_stop();
                    return;
                }
            }
        }
        // Shadow-model check of the previous insn's effect on dst8.
        if let (true, Some(d)) = (s.check, s.dst8) {
            if let Some(actual) = read_reg(emu, d, bits).map(|v| v as u8) {
                if actual != s.pred {
                    if let Some((pva, ptxt)) = s.prev.clone() {
                        s.unmodeled.push(format!("{:#x}: {}", pva, ptxt));
                    }
                    s.pred = actual;
                }
            }
        }
        s.last_pc = addr;
        let ins = match decode_at(emu, addr, bits) {
            Some(i) => i,
            None => return,
        };
        s.prev = Some((addr, format!("{}", ins)));
        let Some(d) = s.dst8 else { return; };
        let two_regs = ins.op_count() == 2 && ins.op0_kind() == OpKind::Register
            && ins.op1_kind() == OpKind::Register;
        // Rolling-key update ends the chain.
        if let Some(k) = s.key8 {
            if ins.mnemonic() == Mnemonic::Xor && two_regs && ins.op0_register() == k
                && ins.op1_register() == d
            {
                s.out = read_reg(emu, d, bits).map(|v| v as u8);
                s.key_update_va = Some(addr);
                let _ = emu.emu_stop();
                return;
            }
        }
        if ins.op_count() < 1 || ins.op0_kind() != OpKind::Register || ins.op0_register() != d {
            return;
        }
        if s.key8.is_none() && ins.mnemonic() == Mnemonic::Xor && two_regs
            && ins.op1_register().size() == 1
        {
            let kv = read_reg(emu, ins.op1_register(), bits).unwrap_or(0) as u8;
            s.key8 = Some(ins.op1_register());
            s.key_val = kv;
            s.pred ^= kv;
            return;
        }
        if let Some((op, val)) = model(emu, &ins, bits) {
            if s.key8.is_none() {
                s.unmodeled.push(format!("{:#x}: {} (before key mix)", addr, ins));
            }
            s.ops.push((op, val));
            s.pred = step(op, val, s.pred);
        }
    }).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let res = emu.emu_start(entry, u64::MAX, 0, MAX_INSNS + 10);
    let _ = emu.remove_hook(hr);
    let _ = emu.remove_hook(hc);
    let _ = emu.remove_hook(hw);
    for (addr, old) in undo.borrow().iter().rev() {
        let _ = emu.mem_write(*addr, old);
    }
    for (pg, len) in lazy.borrow_mut().drain(..) {
        let _ = emu.mem_unmap(pg, len);
    }
    let run = Rc::try_unwrap(st).ok().map(|c| c.into_inner())
        .ok_or_else(|| anyhow::anyhow!("hook state still shared"))?;
    if let Err(e) = res {
        if run.out.is_none() {
            anyhow::bail!("emulation error at {:#x}: {:?}", run.last_pc, e);
        }
    }
    Ok(run)
}

/// Mine the fetch decryptor reachable from `entry` (handler entry or
/// fetch site) by emulating all 256 input bytes. `image` comes from
/// [`image_of`]; `bits` is 32 or 64.
pub fn mine_cryptor_emu(image: &Image, entry: u64, bits: u32) -> Result<EmuMined> {
    let (mut emu, lazy) = make_emu(image, bits)?;
    let mut runs = Vec::with_capacity(256);
    for raw in 0..=255u8 {
        runs.push(run_one(&mut emu, &lazy, entry, raw, bits, 0)?);
    }
    let alt = mine_pass_chain(&mut emu, &lazy, entry, bits, 1);
    let r0 = &runs[0];
    let fetch_va = r0.fetch_va.ok_or_else(|| anyhow::anyhow!("no 1-byte bytecode fetch from {:#x}", entry))?;
    // Entry *is* a byte load (started at a fetch site): it must be the
    // fetch. Indexed fetches (`[esi+ecx-X]`) whose index is set before the
    // entry miss the bytecode and would silently mine the next handler.
    if fetch_va != entry {
        if let Some(i) = decode_at(&emu, entry, bits) {
            let byte_load = (0..i.op_count()).any(|k| i.op_kind(k) == OpKind::Memory)
                && i.memory_size().size() == 1
                && i.op_count() == 2 && i.op0_kind() == OpKind::Register;
            if byte_load {
                anyhow::bail!("byte load at entry {:#x} missed the bytecode (address uses state set \
                    before it; start at the handler entry); next fetch was {:#x}", entry, fetch_va);
            }
        }
    }
    let key8 = r0.key8.ok_or_else(|| anyhow::anyhow!("no key mix after fetch {:#x}", fetch_va))?;
    let key_update_va = r0.key_update_va
        .ok_or_else(|| anyhow::anyhow!("no rolling-key update after fetch {:#x}", fetch_va))?;
    for r in &runs[1..] {
        if r.fetch_va != r0.fetch_va || r.key8 != r0.key8 || r.key_update_va != r0.key_update_va {
            anyhow::bail!("control flow depends on the fetched byte (entry {:#x})", entry);
        }
        if r.ops != r0.ops {
            anyhow::bail!("chain operands depend on the fetched byte (entry {:#x})", entry);
        }
    }
    let mut cryptor = ValueCryptor::new(CryptSize::Byte);
    for (op, v) in &r0.ops {
        cryptor.add(*op, *v);
    }
    let table: Vec<u8> = runs.iter().map(|r| r.out.unwrap_or(0)).collect();
    let reproduces = runs.iter().enumerate().all(|(raw, r)| {
        r.out == Some(cryptor.encrypt((raw as u8 ^ r.key_val) as u64) as u8)
    });
    let entry_independent = match alt {
        Some(c) => (0..=255u64).all(|x| c.encrypt(x) == cryptor.encrypt(x)),
        None => false,
    };
    let mut unmodeled: Vec<String> = runs.iter().flat_map(|r| r.unmodeled.clone()).collect();
    unmodeled.sort();
    unmodeled.dedup();
    let name = |r: Register| format!("{:?}", r).to_lowercase();
    Ok(EmuMined {
        entry_va: entry,
        fetch_va,
        vip_reg: r0.vip.map(name).unwrap_or_default(),
        key_reg: name(key8),
        key_update_va,
        verified: reproduces && unmodeled.is_empty() && entry_independent,
        entry_independent,
        cryptor,
        unmodeled,
        table,
    })
}

/// Chain from a perturbed-entry pass (fetch/key/ops consistent across all
/// inputs), or None if that pass fails or disagrees with itself.
fn mine_pass_chain(emu: &mut Unicorn<()>, lazy: &LazyPages, entry: u64, bits: u32, variant: u64) -> Option<ValueCryptor> {
    let mut first: Option<Run> = None;
    for raw in 0..=255u8 {
        let r = run_one(emu, lazy, entry, raw, bits, variant).ok()?;
        r.key_update_va?;
        if let Some(f) = &first {
            if r.ops != f.ops || r.fetch_va != f.fetch_va {
                return None;
            }
        } else {
            first = Some(r);
        }
    }
    let mut c = ValueCryptor::new(CryptSize::Byte);
    for (op, v) in &first?.ops {
        c.add(*op, *v);
    }
    Some(c)
}
