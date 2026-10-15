//! Trace linearization for x86-32 handlers (lift input).
//!
//! VMP 3.x handlers are cut into tiny blocks glued by direct `jmp`s, junk
//! `call`s (push + jump, popped later) and opaque `jcc`s. Lifting blocks
//! one by one loses every cross-block simplification. Given the *executed*
//! path of one handler (live tracer `paths.jsonl`), this re-emits it as one
//! contiguous byte string with identical semantics along that path:
//!
//! | executed insn            | emitted                          |
//! |--------------------------|----------------------------------|
//! | `jmp rel` (direct)       | nothing                          |
//! | `jcc` / `jecxz` (taken or not: the next VA says which) | nothing |
//! | `loop*`                  | `lea ecx,[ecx-1]` (flags untouched) |
//! | `call rel`               | `push <return VA>`               |
//! | `ret` / `ret n` mid-path | `lea esp,[esp+4(+n)]` (flags untouched) |
//! | last insn (dispatch)     | kept as is (`jmp reg/[mem]`, `ret`) |
//! | anything else            | original bytes                   |
//!
//! x86-32 has no RIP-relative operands, so non-branch bytes are position
//! independent. Indirect calls / jumps before the end are refused: the
//! path would leave the handler (native code), not linearizable here.
//! `loope/loopne` also test ZF and are refused.

use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, FlowControl, Mnemonic};

/// One linearized handler path.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Handler entry (first VA of the path); lift at this address.
    pub va: u64,
    pub bytes: Vec<u8>,
    /// Executed insns in the path.
    pub insns_in: usize,
    /// Insns emitted (rewrites count as one).
    pub insns_out: usize,
    /// Direct jmps / jccs dropped.
    pub dropped: usize,
}

/// Linearize an executed x86-32 path. `read` supplies code bytes per VA.
pub fn linearize_path32(path: &[u64], read: &dyn Fn(u64, usize) -> Option<Vec<u8>>) -> Result<Linear> {
    anyhow::ensure!(!path.is_empty(), "empty path");
    let mut out = Vec::new();
    let (mut kept, mut dropped) = (0usize, 0usize);
    for (i, &va) in path.iter().enumerate() {
        let code = read(va, 16).ok_or_else(|| anyhow::anyhow!("no bytes at {:#x}", va))?;
        let mut d = Decoder::with_ip(32, &code, va, DecoderOptions::NONE);
        let ins = d.decode();
        anyhow::ensure!(!ins.is_invalid(), "invalid insn at {:#x}", va);
        let raw = &code[..ins.len()];
        let last = i + 1 == path.len();
        let next = path.get(i + 1).copied();
        if last {
            out.extend_from_slice(raw);
            kept += 1;
            break;
        }
        match ins.flow_control() {
            FlowControl::Next => {
                out.extend_from_slice(raw);
                kept += 1;
            }
            FlowControl::UnconditionalBranch => {
                anyhow::ensure!(next == Some(ins.near_branch_target()),
                    "jmp at {:#x} does not reach next path VA", va);
                dropped += 1;
            }
            FlowControl::ConditionalBranch => match ins.mnemonic() {
                Mnemonic::Loop => {
                    out.extend_from_slice(&[0x8D, 0x49, 0xFF]); // lea ecx,[ecx-1]
                    kept += 1;
                }
                Mnemonic::Loope | Mnemonic::Loopne => {
                    anyhow::bail!("{:?} at {:#x} not linearizable", ins.mnemonic(), va)
                }
                _ => dropped += 1, // jcc / jecxz: path already chose
            },
            FlowControl::Call => {
                anyhow::ensure!(next == Some(ins.near_branch_target()),
                    "call at {:#x} does not reach next path VA", va);
                out.push(0x68); // push imm32 (return address)
                out.extend_from_slice(&(ins.next_ip() as u32).to_le_bytes());
                kept += 1;
            }
            FlowControl::Return => {
                let n = if ins.op_count() == 1 { ins.immediate16() as u32 } else { 0 };
                out.extend_from_slice(&[0x8D, 0xA4, 0x24]); // lea esp,[esp+disp32]
                out.extend_from_slice(&(4 + n).to_le_bytes());
                kept += 1;
            }
            other => anyhow::bail!("{:?} ({}) at {:#x} before the end of the path", other, ins, va),
        }
    }
    Ok(Linear { va: path[0], bytes: out, insns_in: path.len(), insns_out: kept, dropped })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader(img: &[(u64, Vec<u8>)]) -> impl Fn(u64, usize) -> Option<Vec<u8>> + '_ {
        move |va, n| {
            for (b, d) in img {
                if va >= *b && va < *b + d.len() as u64 {
                    let o = (va - b) as usize;
                    return Some(d[o..(o + n).min(d.len())].to_vec());
                }
            }
            None
        }
    }

    #[test]
    fn glues_jmp_call_ret_and_keeps_dispatch() {
        // 0x1000: xor al,bl ; jmp 0x2000
        // 0x2000: call 0x3000
        // 0x3000: inc al ; ret            (returns to 0x2005)
        // 0x2005: jz 0x2009 (not taken) ; 0x2007: jmp ebp
        let img = vec![
            (0x1000u64, vec![0x30, 0xD8, 0xE9, 0xF9, 0x0F, 0x00, 0x00]),
            (0x2000u64, vec![0xE8, 0xFB, 0x0F, 0x00, 0x00, 0x74, 0x02, 0xFF, 0xE5]),
            (0x3000u64, vec![0xFE, 0xC0, 0xC3]),
        ];
        let path = [0x1000, 0x1002, 0x2000, 0x3000, 0x3002, 0x2005, 0x2007];
        let l = linearize_path32(&path, &reader(&img)).unwrap();
        assert_eq!(l.bytes, vec![
            0x30, 0xD8,                         // xor al,bl
            0x68, 0x05, 0x20, 0x00, 0x00,       // push 0x2005
            0xFE, 0xC0,                         // inc al
            0x8D, 0xA4, 0x24, 0x04, 0, 0, 0,    // lea esp,[esp+4]
            0xFF, 0xE5,                         // jmp ebp
        ]);
        assert_eq!((l.insns_in, l.insns_out, l.dropped), (7, 5, 2));
    }

    #[test]
    fn refuses_jmp_that_leaves_the_path() {
        let img = vec![(0x1000u64, vec![0xEB, 0x10, 0xFF, 0xE5])]; // jmp +0x10
        assert!(linearize_path32(&[0x1000, 0x1002], &reader(&img)).is_err());
    }
}
