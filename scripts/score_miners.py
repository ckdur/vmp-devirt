"""Score fetch-chain miners against an emulation oracle (fetch_oracle32.py).

Modes: static `devirt mine` at the oracle fetch site; `devirt mine-emu`
from the handler entry and from the fetch site. A chain is right when
chain(raw ^ key[raw]) == out[raw] for all 256 oracle inputs.
"verified" = mine-emu's own claim; "MISMATCH verified" is a silent error.

usage: score_miners.py <devirt> <binary> <oracle.json>
"""
import collections
import json
import re
import subprocess
import sys


def apply(op, val, x):
    if op == "Inc": return (x + 1) & 0xff
    if op == "Dec": return (x - 1) & 0xff
    if op == "Not": return ~x & 0xff
    if op == "Neg": return -x & 0xff
    if op == "Xor": return x ^ val
    if op == "Add": return (x + val) & 0xff
    if op == "Sub": return (x - val) & 0xff
    if op == "And": return x & val
    n = (val & 31) % 8
    if op == "Rol": return ((x << n) | (x >> (8 - n))) & 0xff
    if op == "Ror": return ((x >> n) | (x << (8 - n))) & 0xff
    raise ValueError(op)


def matches(chain_text, v):
    cmds = re.findall(r"op: (\w+), size: \w+, value: (\d+)", chain_text)
    for raw in range(256):
        x = raw ^ v["keys"][raw]
        for op, val in cmds:
            x = apply(op, int(val), x)
        if x != v["table"][raw]:
            return False
    return True


def bucket(err):
    for pat, name in (("byte load at entry", "missed fetch"), ("emulation error", "emulation error"),
                      ("no ", "no chain"), ("depend", "no chain")):
        if pat in err:
            return "error: " + name
    return "error: other"


devirt, binary, oracle = sys.argv[1:4]
d = {h: v for h, v in json.load(open(oracle)).items() if "table" in v}

# static miner, per distinct fetch site
sites = {v["fetch"]: v for v in d.values()}
st = collections.Counter()
for va, v in sites.items():
    out = subprocess.run([devirt, "mine", binary, va], capture_output=True, text=True).stdout
    m = re.search(r"key=(\S+) .*steps=\d+ (\[.*\])", out)
    if not m:
        st["no chain"] += 1
    else:
        st["match" if m.group(1) == v["key"] and matches(m.group(2), v) else "MISMATCH"] += 1
print("static mine @fetch  sites=%d %s" % (len(sites), dict(st)))

for label, pick in (("mine-emu @entry", lambda h, v: h), ("mine-emu @fetch", lambda h, v: v["fetch"])):
    vas = sorted({pick(h, v) for h, v in d.items()})
    out = subprocess.run([devirt, "mine-emu", binary] + vas, capture_output=True, text=True).stdout
    res = dict(re.findall(r"^entry (0x[0-9a-f]+) (.*)$", out, re.M))
    st = collections.Counter()
    for h, v in d.items():
        r = res.get(pick(h, v), "error: missing")
        if r.startswith("error"):
            st[bucket(r)] += 1
            continue
        ok = matches(r, v)
        st[("match" if ok else "MISMATCH") + (" verified" if "verified=true" in r else " unverified")] += 1
    print("%-19s handlers=%d %s" % (label, len(d), dict(st)))
