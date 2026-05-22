# ADR: Feature Flag Simplification — CANN Always, Simpler Optional

**Date:** 2026-05-22
**Status:** Proposed

## Context

### Project Identity

The Serving Agent is an **Ascend NPU-only** project. It has no CUDA, ROCm, or
other hardware backend. The entire inference pipeline — model loading, operator
execution, KV cache management — is built on Huawei's CANN SDK
(`libascendcl.so`, `libopapi.so`, `libhccl.so`). There is no CPU inference
path in production.

Despite this single-platform identity, the Cargo feature system encodes a
multi-platform illusion that has grown complex and brittle.

### Current Crate Dependency Graph

```
workspace (serving_agent/)
├── rust_llm_server          (the binary + library crate)
│   ├── ascend               (safe wrappers, OPTIONAL via "ascend" feature)
│   │   ├── ascendcl-sys     (raw AscendCL FFI, OPTIONAL)
│   │   ├── aclnn-sys        (raw aclnn FFI, OPTIONAL)
│   │   └── hccl-sys         (raw HCCL FFI, OPTIONAL via "hccl" feature)
│   ├── simpler-sys          (raw simpler FFI, OPTIONAL via "ascend" feature)
│   └── kv-cache             (pure Rust, unconditional)
```

Every `ascend`-related dep is `optional = true`, gated behind the crate-level
`ascend` feature. This means the entire CANN binding stack can be compiled out
with a single flag decision.

### Feature Inventory

| Feature | Defined in | Gates |
|---------|-----------|-------|
| `ascend` | `rust_llm_server/Cargo.toml` | Pulls in `ascend`, `ascendcl-sys`, `aclnn-sys`, `simpler-sys` as deps; 64 `#[cfg(feature = "ascend")]` sites in source, plus 7 `#[cfg(not(feature = "ascend"))]` / `#[cfg_attr(...)]` sites |
| `hccl` | `rust_llm_server/Cargo.toml` | Pulls in `hccl-sys`, enables `ascend/hccl`; 18 `#[cfg(all(feature = "ascend", feature = "hccl"))]` sites and 1 `#[cfg(all(feature = "ascend", not(feature = "hccl")))]` stub module in `ops/mod.rs` |
| `stub` | `ascendcl-sys`, `aclnn-sys`, `hccl-sys`, `simpler-sys`, `ascend` Cargo.tomls | Skips linking in 3 build.rs files; provides panic-on-use stub impls in `simpler-sys/src/stub.rs`; 18 `#[cfg(feature = "stub")]` sites in `simpler-sys` |
| `pool_depth_N` | `rust_llm_server/Cargo.toml` | Compile-time scratch arena pool depth (orthogonal) |

**Total cfg annotations in play: ~92** (64 positive ascend + 7 negative ascend
+ 18 hccl-combined + 1 hccl-stub fallback + 2 `#[cfg_attr(not(feature = "hccl"), ...)]`).

### What Each Feature Actually Does at the Code Level

**`ascend` feature** (64 `#[cfg(feature = "ascend")]` sites across 10 files):

| File | Sites | What it gates |
|------|-------|---------------|
| `engine.rs` | 22 | 6 struct fields (`ascend_ops`, `comm_ops`, `acl_context`, `weight_tensors_v2`, `kv_key_caches`, `kv_value_caches`, `decode_buffers`); `Engine::new_ascend()` constructor; `generate()`, `run_forward_step()`, `compute_stream()`, `set_comm_ops()`, `run_worker_loop()` methods; perf timing imports |
| `device_tensor.rs` | 11 | Entire file (`DeviceTensor`, `WeightTensor` types, upload/download functions) |
| `plan.rs` | 10 | Device execution step types (`ExecStep::Matmul`, `ExecStep::RmsNorm`, etc.) and device plan compilation |
| `scratch_arena.rs` | 7 | Device-side scratch buffer pool, `RotatingPool` |
| `main.rs` | 5 | `AscendComputeOps::new()` init, `upload_weights_to_device()`, device ID resolution chain |
| `tensor.rs` | 4 | `DeviceTensorRef` type, `as_device_tensor()`, `device_buf()`, `data_ptr()` |
| `debug_dump.rs` | 2 | Device memory debug dump functions |
| `lib.rs` | 1 | `use simpler_sys as _;` |
| `ops/mod.rs` | 1 | `pub mod ascend; pub mod ascend_comm;` |
| `weights.rs` | 1 | `upload_weights_to_device()` function |

In addition to the 64 positive sites, there are 7 **negative** /
`#[cfg_attr]` sites that also need removal:

| File | Sites | What it gates |
|------|-------|---------------|
| `main.rs` | 3 | `#[cfg(not(feature = "ascend"))]` error branch for "ascend backend requested but not compiled"; `#[cfg_attr(not(feature = "ascend"), ...)]` default value for `--backend`; `#[cfg(not(feature = "ascend"))]` stub `Engine::new()` construction path |
| `lib.rs` | 2 | `#[cfg_attr(not(feature = "ascend"), allow(unused_imports))]` on `half` and `tracing_subscriber` |
| `engine.rs` | 2 | `#[cfg(not(feature = "ascend"))]` fallback `generate()` returning empty; `#[cfg_attr(not(all(feature = "ascend", feature = "hccl")), allow(unused_mut))]` |

**`hccl` feature** (appears in combination with `ascend`):

| File | Sites | What it gates |
|------|-------|---------------|
| `ascend/src/lib.rs` | 1 | `pub mod comm;` — the entire HCCL communicator wrapper module |
| `rust_llm_server/src/ops/mod.rs` | 2 | `pub mod ascend_comm;` (real) vs. stub `ascend_comm` module with panic-on-use methods |
| `rust_llm_server/src/engine/engine.rs` | 9 | Broadcast operations inside `generate()` and `run_forward_step()`; `generate_streaming()` worker loop; `broadcast_paged_inputs()` and `process_group`-related methods |
| `rust_llm_server/src/main.rs` | 6 | `AscendCommOps` import; `process_group` import; HCCL process group init block; `comm_ops` injection via `engine.set_comm_ops()` |
| `rust_llm_server/src/distributed/mod.rs` | 1 | `pub mod process_group;` |
| `PARAL.md` | 1 | Documentation reference (non-code, informational only) |

Two additional `#[cfg_attr(not(feature = "hccl"), allow(unused_mut))]` sites
exist in `engine.rs` and `main.rs` to suppress warnings in single-device
builds.

**`stub` feature** (in 4 sys crates + `simpler-sys/src/`):

| Site | What it gates |
|------|---------------|
| `ascendcl-sys/build.rs` | Early return — skip linking `libascendcl.so` |
| `aclnn-sys/build.rs` | Early return — skip linking `libopapi.so` |
| `hccl-sys/build.rs` | Early return — skip linking `libhccl.so`, `libhcomm.so` |
| `simpler-sys/src/lib.rs` | `pub mod stub;` vs `mod runtime;` — which module to compile |
| `simpler-sys/src/args.rs` | 16 `#[cfg(feature = "stub")]` sites — conditional test logic that skips NPU-dependent assertions |

The `ascend` crate also has `stub = ["ascendcl-sys/stub", "aclnn-sys/stub"]` in
its Cargo.toml, forwarding the stub flag to sub-crates. The `ascend` crate
itself has no `#[cfg(feature = "stub")]` in its source code.

### Current CI Matrix

From `scripts/check`:

1. `cargo check --workspace` → default (no features) → "stub" mode
2. `cargo check -p rust_llm_server --features ascend` → NPU, no HCCL
3. `cargo check -p rust_llm_server --features ascend,hccl` → NPU + multi-device
4. `cargo check -p ascend --features stub` → ascend crate in CI mode

Plus `--examples` variants for combinations 2-3. Total: **4 compilation modes**
that must all pass.

## The Proposal

**Admit what the project is: an Ascend NPU server that always uses CANN.**

Replace the three-way feature matrix (`ascend` / `stub` / `hccl`) with a simple
two-tier model:

```
Always compiled (CANN SDK — required):
  ascendcl-sys ──→ aclnn-sys ──→ hccl-sys ──→ ascend
                                                  │
                                      rust_llm_server
                                      (always Ascend, always linked)

Optional (feature-gated):
  simpler-sys ──→ only compiled with --features simpler
```

### Principles

1. **CANN is required.** Every developer and CI machine must have CANN SDK
   installed (`ASCEND_HOME_PATH`). The `ascend` feature is removed — all
   CANN crates are unconditional dependencies.

2. **NPU hardware is optional.** Compilation succeeds on any machine with
   CANN headers and `.so` files. Running NPU tests or inference needs
   physical NPU cards. This is a **runtime check** (`require_device()`),
   not a compile-time gate. The `--backend stub` path (passing `None` for
   `ascend_ops`) remains available for CPU-only testing of the server
   pipeline.

3. **`simpler-sys` is optional.** The PyPTO kernel dispatch runtime requires
   the simpler compiler infrastructure (C++17 headers for bindgen, the
   `libhost_runtime.so` build output, compiled kernel blobs). This
   experimental integration is gated behind `--features simpler`.

### What changes

| Current | → After |
|---------|---------|
| `ascend` feature in `rust_llm_server/Cargo.toml` | **Removed** — deps are unconditional |
| `hccl` feature in `rust_llm_server/Cargo.toml` | **Removed** — HCCL is part of CANN SDK |
| `[features]` block (except `pool_depth_N` + `simpler`) | **Removed** — 2 features remain (from 5) |
| 64 `#[cfg(feature = "ascend")]` + 7 negative/`cfg_attr` sites | **Removed** — code is always compiled |
| 18 `#[cfg(all(feature = "ascend", feature = "hccl"))]` sites in `rust_llm_server/` | **Removed** — HCCL always compiled |
| 1 `#[cfg(all(feature = "ascend", not(feature = "hccl")))]` stub `ascend_comm` module in `ops/mod.rs` | **Deleted** — stub no longer needed |
| `#[cfg(feature = "hccl")]` in `ascend/src/lib.rs` | **Removed** — `pub mod comm` is unconditional |
| `#[cfg(feature = "stub")]` 18 sites in `simpler-sys/` | **Removed** — stub module deleted |
| `simpler-sys/src/stub.rs` | **Deleted** — no longer needed |
| `stub` feature in 4 sys crate Cargo.tomls | **Removed** — no stub mode |
| `cfg!(feature = "stub")` in 3 build.rs files | **Removed** — build.rs always links |
| `Engine::new()` + `Engine::new_ascend()` dual constructors | **Merged** — single `Engine::new()` taking `Option<AscendComputeOps>` |
| 6+ struct fields in `Engine` behind `#[cfg]` | **Made unconditional** — `ascend_ops` becomes `Option` |
| `--backend` CLI arg with `#[cfg_attr]` default-value tricks and "not compiled" error branch | **Simplified** — pure runtime switch: `stub` → `None`, `ascend` → `Some(ops)`. Default is `ascend`. |
| `required-features` on `[[example]]` targets | **Removed** — no feature gates on targets |
| CI matrix: 4 combinations | **Reduced to 2**: `cargo check` + `cargo check --features simpler` |

### What stays

| Feature | Reason |
|---------|--------|
| `pool_depth_2/3/4` | Compile-time config for scratch arena (orthogonal to hardware) |
| `simpler` (new name) | Optional dep for PyPTO kernel dispatch |

### Cargo.toml (after)

```toml
# rust_llm_server/Cargo.toml
[dependencies]
ascend = { path = "../rustBindings/ascend" }
kv-cache = { path = "../kv-cache" }
simpler-sys = { path = "../rustBindings/simpler-sys", optional = true }
# ... axum, tokio, clap, etc. unchanged ...

[features]
simpler = ["dep:simpler-sys"]
pool_depth_2 = []
pool_depth_3 = []
pool_depth_4 = []
```

```toml
# rustBindings/ascend/Cargo.toml
[dependencies]
ascendcl-sys = { path = "../ascendcl-sys" }
aclnn-sys = { path = "../aclnn-sys" }
hccl-sys = { path = "../hccl-sys" }  # unconditional now
```

The four sys crates (`ascendcl-sys`, `aclnn-sys`, `hccl-sys`, `simpler-sys`)
have their `[features] stub = []` entries removed. Their build.rs files no
longer check for `cfg!(feature = "stub")` — they always try to link.

### Engine Constructor Simplification

Currently there are two constructors. The "stub" one (`Engine::new()`) skips
all Ascend-related field initialization and is used only by CPU unit tests.
The "ascend" one (`Engine::new_ascend()`) initializes NPU resources.

After the change, there is one constructor, and the optionality of NPU is
expressed through `Option` rather than `#[cfg]`:

```rust
pub struct Engine {
    // These were behind #[cfg(feature = "ascend")] — now unconditional
    ascend_ops: Option<AscendComputeOps>,
    comm_ops: Option<AscendCommOps>,         // None for single-device
    acl_context: Option<AclContext>,
    weight_tensors_v2: Vec<WeightTensor>,
    kv_key_caches: Vec<DeviceBuffer>,
    kv_value_caches: Vec<DeviceBuffer>,
    decode_buffers: Mutex<Option<DecodeBuffers>>,
    // ... non-conditional fields unchanged ...
}

impl Engine {
    /// Single constructor: production passes Some(ascend_ops),
    /// unit tests and --backend stub pass None.
    pub fn new(
        model: Qwen3Model,
        ascend_ops: Option<AscendComputeOps>,
        parallel: ParallelConfig,
        quant: QuantConfig,
    ) -> Self { ... }
}
```

Unit tests pass `None` and only test model info, plan compilation, and other
CPU-logic. Production code (`main.rs`) passes `Some(ascend_ops)`. The
`--backend stub` CLI path also passes `None` and exercises the full binary
pipeline (tokenizer → plan → server) without NPU hardware.

### `--backend` CLI Simplification

Today the `--backend` argument uses `#[cfg_attr]` to set different defaults
depending on feature flags and has a compile-time error branch when `ascend`
is requested but not compiled:

```rust
// Before: feature-gated default value
#[cfg_attr(not(feature = "ascend"), arg(long, default_value = "stub"))]
#[cfg_attr(feature = "ascend", arg(long, default_value = "ascend"))]
backend: String,
```

After the change, both `ascend` and `stub` paths are always compiled — no
`#[cfg]` tricks needed:

```rust
// After: simple default, no cfg_attr
#[arg(long, default_value = "ascend")]
backend: String,
```

The resolution logic becomes a pure runtime switch:

```rust
let ascend_ops_init = match cli.backend.as_str() {
    "stub" => {
        tracing::info!("Using STUB backend (no-op operators)");
        None
    }
    "ascend" => {
        let device_id = resolve_device_id(&cli, &distributed)?;
        tracing::info!("Using ASCEND NPU backend (device {})", device_id);
        let ops = AscendComputeOps::new(device_id)
            .map_err(|e| format!("Failed to init Ascend backend: {}", e))?;
        Some(ops)
    }
    other => return Err(format!("Unknown backend: {other}. Use stub or ascend.").into()),
};
```

No "backend not compiled" error branch. No `#[cfg]` on the match arms. Both
paths compile and link — `stub` simply never calls CANN at runtime.

### Build Script Changes

`ascendcl-sys/build.rs`, `aclnn-sys/build.rs`, `hccl-sys/build.rs`:

```rust
// Before (stub branch exists):
let ascend_home_path = match std::env::var("ASCEND_HOME_PATH") {
    Ok(home) => home,
    Err(_) if cfg!(feature = "stub") => return,   // ← removed
    Err(_) => panic!("CANN SDK required..."),
};

// After (always required):
let ascend_home_path = std::env::var("ASCEND_HOME_PATH")
    .expect("CANN SDK required: set ASCEND_HOME_PATH");
```

`simpler-sys/build.rs`: no change needed. Since `simpler-sys` is only compiled
when `--features simpler` is used, its build.rs will fail with a clear error
if the simpler source headers are missing. This is correct behaviour when
you've opted into PyPTO support.

The `simpler-sys/src/stub.rs` module and all `#[cfg(feature = "stub")]` blocks
in `simpler-sys/src/lib.rs` and `simpler-sys/src/args.rs` are deleted.

### Test / Example / Binary Convention

| Category | Mechanism | How to run | Hardware needed? |
|----------|-----------|-----------|:---:|
| CPU unit test | `#[cfg(test)] mod tests` inside `src/` | `cargo test` | No |
| NPU integration test | `tests/npu_*.rs` with `require_device()` | `TASK_DEVICE=x cargo test` | Yes (skipped otherwise) |
| simpler integration test | `tests/npu_simpler.rs` with `#![cfg(feature = "simpler")]` | `cargo test --features simpler` | Yes |
| Example | `examples/*.rs` | `cargo run --example` | Depends on example |
| Binary | `main.rs` | `cargo run -- --weights ...` | No (with `--backend stub`), Yes (with `--backend ascend`) |

**Runtime skip pattern** — single convention, applied consistently:

```rust
// tests/npu_ops.rs — compiled always, runs only with NPU hardware
fn require_device() -> Option<Device> {
    let id = std::env::var("TASK_DEVICE").ok()?.parse().ok()?;
    Device::init(id).ok()
}

#[test]
fn test_matmul_small() {
    let Some(_dev) = require_device() else { return };
    // ... actual NPU test ...
}
```

```rust
// tests/npu_simpler.rs — compiled only when opted in
#![cfg(feature = "simpler")]

#[test]
fn test_hello_world_kernel() {
    let device_id = std::env::var("TASK_DEVICE").expect("TASK_DEVICE");
    // ... test using simpler kernel dispatch ...
}
```

No `#[ignore]`, no `required-features`, no conditional compilation within
function bodies (except the file-level `#![cfg]` for the `simpler` feature).

### Scripts

```bash
scripts/test-cpu      → cargo test --workspace
scripts/test-npu      → task-submit --device auto --run "cargo test"
scripts/test-simpler  → task-submit --device auto --run "cargo test --features simpler"
scripts/check         → fmt + clippy + doc + test-no-run for workspace
scripts/run           → task-submit --device auto --run \
                          "cargo run -- --weights $MODEL_DIR --prompt '...'"
```

Scripts that disappear:

| Current script | Why it goes away |
|---------------|-----------------|
| `scripts/test-smoke` | Absorbed into `cargo test` — no special script needed for simpler-sys smoke test |
| `scripts/test-integration` | Absorbed into `cargo test` — ascend integration tests are just regular tests |
| `scripts/hccl-smoke` | **Kept** — multi-rank testing requires a separate launcher (2-rank `task-submit`). Not absorbable into single-process `cargo test`. |
| `scripts/hccl-cpp-test` | **Kept** — separate C++ testing concern, not affected by this ADR |

## Rationale

### Why remove the `ascend` feature?

Every `#[cfg(feature = "ascend")]` asks the same question: "Does this function
exist?" But the answer is always "yes" on every machine where this project is
developed. The `ascend` feature is a proxy for "Do we have CANN?" — and the
answer to that is not a per-crate decision, but a project-wide invariant.

Removing ~92 `#[cfg]` sites eliminates an entire class of bugs (forgetting to
gate a function, gating the wrong function, unused-import warnings because a
dependency is only used behind a feature gate). It makes the code **read as
what it is**: an Ascend NPU server.

The remaining distinction — "do we have an NPU card right now?" — is expressed
honestly through `Option<AscendComputeOps>` at runtime, not through
compile-time feature gates.

### Why remove the `stub` feature?

Three of the four sys crates (`ascendcl-sys`, `aclnn-sys`, `hccl-sys`) use
`stub` to skip **linking**, not compilation. Their Rust source code is fully
compiled either way. The only thing `stub` does is avoid
`rustc-link-lib=dylib=ascendcl` in the build script — but the Rust type
definitions and function declarations are still there.

This means CI machines already need CANN headers installed to compile these
crates. Once headers are present, there is no cost to linking — `.so` files
are loaded lazily by the dynamic linker.

For `simpler-sys`, the `stub` feature provides panic-on-use implementations
(via `simpler-sys/src/stub.rs`). With `simpler-sys` becoming optional, this
module is no longer needed — if you don't enable `--features simpler`, the
crate isn't compiled at all.

### Why remove the `hccl` feature?

`libhccl.so` and `libhcomm.so` live in `$ASCEND_HOME/lib64/` — the same
directory as `libascendcl.so`. Every CANN installation includes HCCL. There
is no scenario where "I have CANN but not HCCL."

The `ascend::comm` module (HCCL wrappers for `HcclCommunicator`,
`AllReduce`, `Broadcast`) is ~300 lines of safe Rust. Conditionalizing it
adds 18 `#[cfg]` sites across 5 files for zero practical benefit. The stub
`ascend_comm` module in `ops/mod.rs` (a panic-on-use fallback for builds
without HCCL) becomes dead code and is deleted.

### Why keep `simpler` optional?

The simpler runtime is a separate project with its own build pipeline:
1. C++17 headers needed by `bindgen` during compilation
2. The `libhost_runtime.so` shared library (must be pre-built)
3. Compiled kernel blobs (`.bin` + `.json` files from the pypto compiler)
4. A custom `DeviceContext` init path

Developers working on the inference engine (attention kernels, KV cache, HTTP
server, tokenizer) don't need simpler. Making it optional avoids forcing them
to install an additional toolchain. The feature also serves as a clear signal:
"this crate is experimental, opt in explicitly."

## Consequences

### Positive

- **~92 `#[cfg]` annotations removed** from the codebase — less mental overhead,
  fewer compilation modes to reason about.
- **Engine struct simplified** — no feature-gated fields, single constructor
  with `Option<AscendComputeOps>`.
- **`ops/mod.rs` stub `ascend_comm` module deleted** — single real implementation,
  no fallback.
- **CI matrix halved** — from 4 combinations to 2.
- **Tests become regular tests** — no `required-features`, no `#[ignore]`,
  one runtime skip convention (`require_device()`).
- **Scripts consolidated** — test-smoke and test-integration absorbed into
  `cargo test`.
- **`--backend` simplified** — no `#[cfg_attr]` default-value tricks, no
  "backend not compiled" error branch. Pure runtime switch. `stub` remains
  available for CPU-only pipeline testing.
- **`Cargo.toml` shrinks** — 5 features → 2.

### Negative

- **CANN SDK becomes a hard build dependency.** Developers without CANN cannot
  compile the project. Currently, `cargo check --workspace` (no features) works
  without CANN. After the change, all compilation paths require
  `ASCEND_HOME_PATH` to be set. This is a trade-off: we accept it because the
  project has no other deployment target and CI machines already have CANN
  installed for the `--features ascend` check modes.
- **`cargo test --workspace` now requires CANN SDK installed** even for CPU-only
  unit tests. Tests that don't need NPU will compile and run successfully
  (linking against CANN `.so` files but never calling them at runtime). If a
  developer is on a machine without CANN, they cannot compile the project at
  all.

### Neutrals

- **simpler-sys test organization.** With simpler as an optional dependency,
  its tests live behind `#![cfg(feature = "simpler")]`. This is slightly more
  complex than unconditional tests, but isolates the experimental integration
  cleanly.
- **`Option<AscendComputeOps>` in the Engine.** Unit tests and `--backend stub`
  pass `None`, production passes `Some(...)`. This is a runtime check instead
  of a compile-time guarantee, but it's honest — even in production,
  `ascend_ops` could fail to initialize (no NPU card, out of memory).

## Open Questions

### `simpler` feature activation

When `--features simpler` is enabled, does `rust_llm_server` need to *do*
anything differently at runtime, or is it purely a "compile the crate so
we can link it later" flag? Currently, the library only does
`use simpler_sys as _;` — no functional integration exists yet. The feature
may need to also enable runtime code paths when those are implemented.

### Test directory layout

Should NPU tests live alongside the crate they test
(`ascend/tests/npu_ops.rs`, `rust_llm_server/tests/npu_inference.rs`), or
in a workspace-level `tests/` directory? This affects discoverability and
`cargo test -p <crate>` invocations. Proposed: colocate with the crate
(keep current structure).

### Multi-rank HCCL testing

Currently `scripts/hccl-smoke` launches a 2-rank multi-process test via
`task-submit --device-num 2`. Merging this into `cargo test` would require
either spawning child processes from within a Rust test or running `cargo test`
under a multi-rank launcher — neither is straightforward. The HCCL smoke script
is kept as-is; a future ADR should address multi-rank test infrastructure.

## Migration Plan

### Phase 1: Build infrastructure (1-2 days)

1. Remove `stub = []` from `[features]` in all 4 sys crate Cargo.tomls
   (`ascendcl-sys`, `aclnn-sys`, `hccl-sys`, `simpler-sys`).
2. Remove `cfg!(feature = "stub")` early-return from 3 build.rs files
   (`ascendcl-sys`, `aclnn-sys`, `hccl-sys`). Change to unconditional
   `expect()`.
3. Delete `simpler-sys/src/stub.rs`.
4. Remove all `#[cfg(feature = "stub")]` from `simpler-sys/src/lib.rs`
   and `simpler-sys/src/args.rs`.

### Phase 2: Dependency simplification (2-3 days)

5. In `rust_llm_server/Cargo.toml`:
   - Move `ascend`, `ascendcl-sys`, `aclnn-sys`, `hccl-sys` from
     `[dependencies]` (optional) to unconditional.
   - Remove `ascend = [...]` from `[features]`.
   - Remove `hccl = [...]` from `[features]`.
   - Add `simpler = ["dep:simpler-sys"]`.
6. In `rustBindings/ascend/Cargo.toml`:
   - Move `hccl-sys` from optional to unconditional.
   - Remove `stub = [...]` and `hccl = [...]` from `[features]`.
7. Verify `cargo build` works on a machine with CANN SDK.

### Phase 3: Code cleanup (3-5 days)

8. Remove all `#[cfg(feature = "ascend")]` annotations from:
   - `rust_llm_server/src/lib.rs` (1 site)
   - `rust_llm_server/src/ops/mod.rs` (1 site)
   - `rust_llm_server/src/model/tensor.rs` (4 sites)
   - `rust_llm_server/src/model/device_tensor.rs` (11 sites)
   - `rust_llm_server/src/model/weights.rs` (1 site)
   - `rust_llm_server/src/model/scratch_arena.rs` (7 sites)
   - `rust_llm_server/src/engine/plan.rs` (10 sites)
   - `rust_llm_server/src/engine/engine.rs` (22 sites)
   - `rust_llm_server/src/engine/debug_dump.rs` (2 sites)
   - `rust_llm_server/src/main.rs` (5 sites)
9. Remove all negative / `#[cfg(not(feature = "ascend"))]` / `#[cfg_attr]` sites:
   - `rust_llm_server/src/main.rs` (3 sites)
   - `rust_llm_server/src/lib.rs` (2 sites)
   - `rust_llm_server/src/engine/engine.rs` (2 sites)
10. Remove all `#[cfg(feature = "hccl")]` and combined HCCL gates:
    - `rustBindings/ascend/src/lib.rs` (1 site)
    - `rust_llm_server/src/ops/mod.rs` (3 sites — delete stub `ascend_comm` module)
    - `rust_llm_server/src/engine/engine.rs` (11 sites)
    - `rust_llm_server/src/main.rs` (8 sites)
    - `rust_llm_server/src/distributed/mod.rs` (1 site)
11. Merge `Engine::new()` and `Engine::new_ascend()` into a single constructor
    that takes `Option<AscendComputeOps>`.
12. Simplify `--backend` CLI arg in `main.rs`:
    - Remove `#[cfg_attr]` default-value tricks.
    - Remove "backend not compiled" error branch.
    - Keep `stub` → `None` and `ascend` → `Some(ops)` as runtime paths.
    - Set default to `"ascend"`.
13. Remove `required-features` from all `[[example]]` targets.

### Phase 4: Test reorganization (1-2 days)

14. NPU integration tests: ensure they use the `require_device()` runtime
    skip pattern. Remove any `#[ignore]` or `#[cfg_attr(..., ignore)]`.
15. Rename or reorganize test files for discoverability (e.g.
    `tests/npu_ops.rs`, `tests/npu_hccl.rs`).
16. Move existing test content from `ascend/tests/integration.rs` and
    `simpler-sys/tests/smoke_test.rs` into the new structure.

### Phase 5: Scripts and documentation (1 day)

17. Remove `scripts/test-smoke`, `scripts/test-integration`.
18. Add `scripts/test-cpu`, `scripts/test-npu`, `scripts/test-simpler` (or
    adapt existing scripts).
19. Simplify `scripts/check` — remove the 4-combination feature matrix.
20. Update `AGENTS.md` to document the new convention.
