# CLAUDE.md

Guidance for Claude Code working in this repo. Read `README.md` (scope,
layout, method) and `docs/protector-policy.md` (no-regression rules)
before non-trivial changes.

## What this is

Trace-assisted VMProtect (1.x/2.x/3.x) + Tigress devirtualization research
in Rust. Protector-specific front-ends (`src/frontend/`) feed one shared
backend (`src/backend/`): Remill lift → LLVM `opt -O3` → semantic cards →
dataflow → native objects. Execution comes from Unicorn snapshots
(`src/harness/snapshot.rs`). Python helpers in `scripts/` (Triton, angr,
Ghidra, replay); C++ replay runtime in `tools/replay/`.

## Build & test

```bash
cargo build --release                       # pure Rust, no LLVM needed
cargo test --release                        # Tier 1: synthetic fixtures only
cargo test --release <name>                 # single test, e.g. demo_runs
```

Optional LLVM link (`--features llvm`, `llvm-sys = "221"`) needs **exactly
LLVM 22** — llvm-sys rejects other majors.

### macOS (this machine)

- Homebrew default `llvm` is 23 (won't satisfy llvm-sys 221). Use the
  keg-only `llvm@22` (`brew install llvm@22`).
- `build.rs` links `libatomic` on Linux only; macOS has no libatomic
  (atomics are in libSystem). Don't make it unconditional again.

```bash
export LLVM_SYS_221_PREFIX=/opt/homebrew/opt/llvm@22
export PATH=/opt/homebrew/opt/llvm@22/bin:$PATH     # `opt` used by llvm_pipeline.rs
export REMILL_LIFT=/Users/ckdur/Documents/PIU/remill/build-llvm17/bin/lift/remill-lift-17
cargo build --release --features llvm
```

### Remill

- Remill is invoked **only as a subprocess** (`lifter.rs::lift_via_remill`:
  `--arch amd64 --bytes <hex> --address <va> --ir_out /dev/stdout`); the
  textual IR is piped to `opt -O3`. No library linking.
- Local Remill lives in `../remill`, built against **LLVM 17**
  (`build-llvm17/`). That is fine: LLVM 22/23 `opt` parses its IR
  (verified). No need to port Remill to LLVM 22.
- `remill_bin()` only auto-probes `remill-lift-22`, `remill-lift`,
  `remill-lift-6.0` on PATH, so `REMILL_LIFT` must be set for the `-17` binary.

## Rules that matter

- **No regressions across protectors** (`docs/protector-policy.md`):
  shared code must keep every number in `tests/golden.json`. No
  `if vmp` / `if tigress` inside backend algorithms — generalize, or put
  the special case in a frontend or behind an env/format gate. Default
  runs without new flags must stay bit-for-bit on the old path.
- **Tier 2 (local, pre-push)**: `scripts/check_golden.py` (needs local
  samples; missing paths skip, exit 1 = regression). Default sample paths
  point at `~/RE/...`; override with `VMP_WORK`, `VMP_DATA`, `TIG_WORK`,
  `DEVIRT`, `REPO_ROOT`. CI cannot run it.
- Sample-gated tests skip silently when fixtures/env (`VMP_TEST_BIN`,
  `VMP_ORACLE`) are absent — a green `cargo test` does not prove sample paths.
- `tests/fixtures/vmp_test.bin` is tracked despite `*.bin` in `.gitignore`;
  never add licensed/commercial samples to the repo.
- Artifacts go to `DATA_DIR` (default `./data`, gitignored); scratch lift
  output to `VMP_WORK_DIR`. See README "Environment" for all env vars.

## Conventions

- Commit subjects: `area: what (measured result)`, e.g.
  `replay: NOHASH hash gating ... (158921/158921, max 200x)`. Common
  areas: `docs`, `replay`, `roadmap`, `tracer`, `fetch`, `miner`,
  `frontend`, `golden`, `scripts`. Report numbers honestly, including
  failures ("honest false").
- Keep module-level `//!` docs current; tools and scripts document their
  inputs in their header comment.
- CLI entry: `src/bin/devirt.rs` (`devirt --help` lists subcommands);
  extra binaries in `tools/` (`trace_open_fetch`, `trace_elf32`,
  `force_edge`, `recompile`).

## x86-32 ELF path (branch `elf32`)

- Format/bitness gates, not protector gates: `PEBinary::decode_bits()`
  returns 64 for every PE (historic, bit-for-bit) and the ELF class for
  ELF. New 32-bit code goes through `*_bits` variants or new functions;
  the old x64 entry points keep their behaviour.
- Section-stripped ELFs (packed loaders) map via PT_LOAD
  (`elf_has_alloc_sections`); ELFs with sections keep the section path
  (Tigress golden depends on it).
- `devirt mine-emu`: start at a handler entry, not a fetch site; trust
  only `verified=true` (shadow model + 256-input reproduction +
  perturbed-entry pass). Ground truth/scoring: `scripts/fetch_oracle32.py`,
  `scripts/score_miners.py`; numbers in `docs/coverage.md`.
- `tools/trace_elf32.rs`: Unicorn i386 tracer with host libc stubs. Long
  loops are expected; keep `LOG=new` (default) so output stays small.
- `devirt lift32`: needs `REMILL_LIFT` (x86 support) plus `opt`/`llc`
  from LLVM 22 on PATH. Default output is the flat-memory analysis view
  (`define_flat_memory` + `wrap_state_local`, function `@handler_<va>`);
  `--opaque-memory` keeps Remill's memory model. The flat view is for
  reading/analysis, not replay.
- Local Remill (LLVM 17 build) emits the x86 sync hyper call (`cpuid`)
  with an all-NUL callee name; `define_pure_intrinsics` renames it.
- `cargo test` covers the ELF32 path with synthetic fixtures only
  (`tests/elf32.rs`, `backend/linearize.rs` tests).
