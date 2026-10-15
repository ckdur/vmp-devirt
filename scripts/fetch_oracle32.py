"""Emulation oracle for VMP x86-32 handler fetch decoders (role-agnostic).

For each executed handler (from deob_out/trace/*.json), run it in Unicorn
256 times from handler entry. Every GPR except ESP points into its own
buffer filled with `raw`, so whichever register is the VIP, its fetch reads
`raw`; low bytes of all pointers are 0; the live key byte at the key mix is
recorded per input (earlier dword fetches may already have rolled it).
  fetch  = first 1-byte read from a buffer (pc + dst8 recorded)
  key    = r8 in the first `xor dst8, r8` after the fetch
  out    = value of X at the rolling-key update `xor key8, X`
Truth: out[raw] == decode(raw, keys[raw]).

usage: fetch_oracle32.py <elf32> <trace.json>... > oracle.json
  trace.json: {"steps": [{"handler": <int VA>}, ...]} (any tracer that
  lists executed handler entries; e.g. a miasm VMP tracer).
deps: unicorn, capstone, pyelftools. Run with python -I.
"""
import json
import sys

from capstone import CS_ARCH_X86, CS_MODE_32, Cs
from elftools.elf.elffile import ELFFile
from unicorn import (UC_ARCH_X86, UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE,
                     UC_HOOK_MEM_READ_UNMAPPED, UC_HOOK_MEM_WRITE_UNMAPPED,
                     UC_MODE_32, Uc, UcError)
from unicorn.x86_const import (UC_X86_REG_EAX, UC_X86_REG_EBP, UC_X86_REG_EBX,
                               UC_X86_REG_ECX, UC_X86_REG_EDI, UC_X86_REG_EDX,
                               UC_X86_REG_EFLAGS, UC_X86_REG_ESI, UC_X86_REG_ESP)

GPR = {"eax": UC_X86_REG_EAX, "ecx": UC_X86_REG_ECX, "edx": UC_X86_REG_EDX,
       "ebx": UC_X86_REG_EBX, "ebp": UC_X86_REG_EBP, "esi": UC_X86_REG_ESI,
       "edi": UC_X86_REG_EDI}
R8 = {"al": ("eax", 0), "cl": ("ecx", 0), "dl": ("edx", 0), "bl": ("ebx", 0),
      "ah": ("eax", 8), "ch": ("ecx", 8), "dh": ("edx", 8), "bh": ("ebx", 8)}
LOW8 = {"eax": "al", "ecx": "cl", "edx": "dl", "ebx": "bl", "ax": "al", "cx": "cl",
        "dx": "dl", "bx": "bl"}
BUF, BUFSZ, STK = 0x60000000, 0x20000, 0x7f000000
LAZY = []

path = sys.argv[1]
segs = [(s["p_vaddr"], s["p_offset"], s["p_filesz"], s["p_memsz"])
        for s in ELFFile(open(path, "rb")).iter_segments() if s["p_type"] == "PT_LOAD"]
data = open(path, "rb").read()
md = Cs(CS_ARCH_X86, CS_MODE_32)

handlers = set()
for tj in sys.argv[2:]:
    for st in json.load(open(tj))["steps"]:
        h = st.get("handler")
        if isinstance(h, int):
            handlers.add(h)


def make_uc():
    uc = Uc(UC_ARCH_X86, UC_MODE_32)
    for va, off, fsz, msz in segs:
        lo, hi = va & ~0xfff, (va + msz + 0xfff) & ~0xfff
        uc.mem_map(lo, hi - lo)
        uc.mem_write(va, data[off:off + fsz])
    uc.mem_map(BUF, BUFSZ * len(GPR))
    uc.mem_map(STK, 0x100000)

    def lazy(uc, acc, addr, size, val, _):
        # Data pages on demand (zeroed, page-wise); code fetches stay fatal.
        for pg in range(addr & ~0xfff, (addr + size + 0xfff) & ~0xfff, 0x1000):
            try:
                uc.mem_read(pg, 1)
                continue
            except UcError:
                pass
            try:
                uc.mem_map(pg, 0x1000)
                LAZY.append(pg)
            except UcError:
                return False
        return True
    uc.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED, lazy)
    return uc


def ops_of(uc, addr, size):
    ins = next(md.disasm(bytes(uc.mem_read(addr, size)), addr), None)
    if ins is None:
        return None, []
    return ins.mnemonic, [o.strip() for o in ins.op_str.split(",")]


def r8_val(uc, r):
    g, sh = R8[r]
    return (uc.reg_read(GPR[g]) >> sh) & 0xff


def run(uc, h, raw):
    """Returns (fetch_va, vip_reg, key8, out_byte)."""
    uc.mem_write(BUF, bytes([raw]) * (BUFSZ * len(GPR)))
    uc.mem_write(STK, b"\0" * 0x100000)
    for i, r in enumerate(GPR.values()):
        uc.reg_write(r, BUF + i * BUFSZ + BUFSZ // 2)
    uc.reg_write(UC_X86_REG_ESP, STK + 0x80000)
    uc.reg_write(UC_X86_REG_EFLAGS, 0x202)
    st = {"fetch": None, "dst": None, "key": None, "out": None, "n": 0, "pc": h, "sz": 1}

    def on_code(uc, addr, size, _):
        st["n"] += 1
        st["pc"], st["sz"] = addr, size
        if st["n"] > 600:
            uc.emu_stop()
            return
        if st["dst"] is None:
            return
        mn, ops = ops_of(uc, addr, size)
        if mn != "xor" or len(ops) != 2 or ops[1] not in R8:
            return
        if st["key"] is None and ops[0] == st["dst"]:
            st["key"] = ops[1]
            st["kval"] = r8_val(uc, ops[1])
        elif st["key"] is not None and ops[0] == st["key"]:
            st["out"] = r8_val(uc, ops[1])
            uc.emu_stop()

    def on_read(uc, acc, addr, size, val, _):
        if st["fetch"] is None and size == 1 and BUF <= addr < BUF + BUFSZ * len(GPR):
            st["fetch"] = st["pc"]
            st["vip"] = list(GPR)[(addr - BUF) // BUFSZ]
            mn, ops = ops_of(uc, st["pc"], st["sz"])
            st["dst"] = LOW8.get(ops[0], ops[0]) if ops else None

    undo = []

    def on_write(uc, acc, addr, size, val, _):
        try:
            undo.append((addr, bytes(uc.mem_read(addr, size))))
        except UcError:
            pass  # lazy page: unmapped after the run

    h1 = uc.hook_add(UC_HOOK_CODE, on_code)
    h2 = uc.hook_add(UC_HOOK_MEM_READ, on_read)
    h3 = uc.hook_add(UC_HOOK_MEM_WRITE, on_write)
    try:
        uc.emu_start(h, 0xffffffff, count=2000)
    finally:
        uc.hook_del(h1)
        uc.hook_del(h2)
        uc.hook_del(h3)
        for addr, old in reversed(undo):  # no leak between inputs
            uc.mem_write(addr, old)
        for pg in LAZY:
            uc.mem_unmap(pg, 0x1000)
        LAZY.clear()
    return st["fetch"], st.get("vip"), st["key"], (st["out"], st.get("kval"))


out = {}
for h in sorted(handlers):
    uc = make_uc()
    try:
        res = [run(uc, h, raw) for raw in range(256)]
    except UcError as e:
        out[hex(h)] = {"error": str(e)}
        continue
    shape = {(f, v, k) for f, v, k, _ in res}
    if len(shape) != 1 or any(o[0] is None for *_, o in res) or None in next(iter(shape)):
        out[hex(h)] = {"error": "no fetch/key-update", "shape": [list(map(str, s)) for s in shape][:3]}
        continue
    f, v, k = shape.pop()
    out[hex(h)] = {"fetch": hex(f), "vip": v, "key": k, "table": [o[0] for *_, o in res],
                  "keys": [o[1] for *_, o in res]}
json.dump(out, sys.stdout, indent=0)
