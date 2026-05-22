# Serving Agent — Rust LLM Inference Server for Ascend NPU

## Layout

```
rust_llm_server/
├── src/
│   ├── main.rs               CLI entry: HTTP server or one-shot generation
│   ├── lib.rs                Crate root
│   ├── engine/
│   │   ├── engine.rs         Inference engine (generate / generate_streaming)
│   │   ├── plan.rs           Compiled execution plan (644 steps for 14B)
│   │   └── kv_cache.rs       Paged KV cache manager
│   ├── model/
│   │   ├── config.rs         Qwen3Config presets: 0.6b, 4b, 8b, 14b
│   │   ├── network.rs        Qwen3Model network graph
│   │   ├── weights.rs        Safetensors weight loader
│   │   ├── tensor.rs         Tensor / DType definitions
│   │   ├── device_tensor.rs  RAII device tensor wrappers
│   │   ├── scratch_arena.rs  Per-layer scratch buffer pool
│   │   ├── quantize.rs       Quantization config
│   │   └── parallel.rs       TP/PP sharding config
│   ├── ops/
│   │   ├── ascend.rs         Ascend NPU compute ops (aclnn)
│   │   └── ascend_comm.rs    HCCL collectives for distributed
│   ├── scheduler/mod.rs      Tokenizer, request/response types, chat template
│   ├── server/mod.rs         Axum HTTP server: /v1/completions, /v1/chat, /health, /v1/models
│   └── distributed/          DistributedConfig, process group init
├── tests/
│   └── npu_ops.rs            CANN operator integration tests (device, memory, matmul, rmsnorm)
└── examples/
    └── hccl_smoke_test.rs    HCCL multi-rank collective smoke test (2 NPU cards)
```

## Feature Flags

The project uses a simplified feature model:

| Feature | Purpose | Default |
|---------|---------|:------:|
| `pool_depth_N` (2/3/4) | Compile-time scratch arena pool depth | No (depth=1) |

**CANN SDK is always required** for compilation (`ASCEND_HOME_PATH` must be set).
NPU hardware is optional at runtime: use `--backend stub` for CPU-only testing.

## Device ID Convention

The project has a strict three-layer rule for NPU device IDs:

| Layer | Rule | Example |
|-------|------|---------|
| **Libraries** (`ascend::Device`, `AscendComputeOps`) | Take `device_id: i32` — no `Option`, no env fallback, no defaults | `Device::init(device_id)` |
| **Tests & Examples** | Use `require_device()` runtime skip pattern — returns `None` if `TASK_DEVICE` unset | `let Some(_) = require_device() else { return };` |
| **Entrypoint** (`main.rs`) | Full resolution chain with logging, graceful error if unset | `--device-id` > `LOCAL_RANK` > `TASK_DEVICE` > `ASCEND_DEVICE_ID` > error |

Tests never silently default to device 0. If `TASK_DEVICE` is missing, tests that
need NPU use `require_device()` and return early. The canonical launcher is
`task-submit --device auto`, which sets `TASK_DEVICE`.

## Testing

Scripts in `scripts/` are the recommended entrance for all testing. They route
work to the right hardware: CPU tests never touch an NPU card, NPU tests only
run the targets that need hardware.

For quick local iteration on CPU-only logic, `cargo test --workspace` is fine —
NPU tests compile but silently skip when `TASK_DEVICE` is not set.

### Test organization

| Location | Kind | Requires NPU | How to run |
|----------|------|:---:|------------|
| `src/**/*.rs` (`#[cfg(test)]` blocks) | CPU unit tests (config, plan, tensor, scheduler, …) | No | `cargo test --workspace` |
| `tests/npu_ops.rs` | CANN operator integration tests (device, memory, matmul, rmsnorm) | Yes | `scripts/test-npu` |
| `examples/hccl_smoke_test.rs` | HCCL 2-rank collective test | Yes (2 cards) | `scripts/test-npu-2` |

### NPU test convention: `require_device()`

Every NPU-dependent test starts with the same skip pattern:

```rust
fn require_device() -> Option<Device> {
    let device_id = std::env::var("TASK_DEVICE").ok()?.parse::<i32>().ok()?;
    Device::init(device_id).ok()
}

#[test]
fn test_something() {
    let Some(_dev) = require_device() else { return };
    // … actual test logic …
}
```

- **No `#[ignore]`**. Silent skip via `require_device()`.
- **No feature flags**. NPU tests are always compiled; runtime gating handles
  the hardware check.
- **`npu_` filename prefix** is a convention for human/script discoverability.

### CANN diagnostic noise

On machines without NPU cards, CANN static initializers may print:

```
DrvMngGetConsoleLogLevel failed. (ret=4)
```

This is **benign**. CANN is always linked and this message is emitted by the
driver stub when no hardware is present. Ignore it.

## Scripts

Scripts in `scripts/` wrap common workflows with correct hardware routing.

| Script | What it runs | Requires NPU |
|--------|-------------|:---:|
| `scripts/check` | All lints: fmt, clippy, doc, test-compile (CI) | No |
| `scripts/test-cpu` | All workspace tests on CPU — NPU tests compile but skip | No |
| `scripts/test-npu` | `cargo test --test 'npu_ops'` via task-submit (1 card) | Yes |
| `scripts/test-npu-2` | HCCL multi-rank example via task-submit (2 cards) | Yes (2 cards) |
| `scripts/hccl-cpp-test` | HCCL C++ 2-rank test | Yes (2 cards) |
| `scripts/run-one-shot` | One-shot generation (reads `MODEL_DIR` from `scripts/env`) | Yes |

### Multi-device scripts

`test-npu-2` and `hccl-cpp-test` accept two optional device arguments
(default: auto-allocate via `task-submit --device-num 2`):

```bash
scripts/test-npu-2        # auto-allocate 2 cards
scripts/test-npu-2 3 5    # use devices 3 and 5 explicitly
```

## Running One-Shot Generation

No HTTP server, just generate and print. CANN is always compiled — no feature
flags needed.

```bash
# Via task-submit (recommended):
scripts/run-one-shot --prompt 'Huawei is' --max-new-tokens 5

# Or directly:
task-submit --device auto --run \
  "cd $(pwd) && cargo run -- \
     --weights /path/to/Qwen3-14B-weights \
     --prompt 'Huawei is' \
     --max-new-tokens 5"
```

Device resolution chain (main.rs entrypoint): `--device-id` > `LOCAL_RANK` (distributed) > `TASK_DEVICE` > `ASCEND_DEVICE_ID` > error.
Tests and examples require `TASK_DEVICE` (set by `task-submit --device auto`) and fail immediately if absent.

## Running the HTTP Server

```bash
task-submit --device auto --run \
  "cd /path/to/serving_agent/rust_llm_server && \
   cargo run -- \
     --weights /path/to/Qwen3-14B-weights"
```

Endpoints: `GET /health`, `POST /v1/completions`, `POST /v1/chat/completions`, `GET /v1/models`.
