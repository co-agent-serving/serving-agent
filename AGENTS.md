# Serving Agent — Rust LLM Inference Server for Ascend NPU

## Layout

```
rust_llm_server/src/
├── main.rs               CLI entry: HTTP server or one-shot generation
├── lib.rs                Crate root
├── engine/
│   ├── engine.rs         Inference engine (generate / generate_streaming)
│   ├── plan.rs           Compiled execution plan (644 steps for 14B)
│   └── kv_cache.rs       Paged KV cache manager
├── model/
│   ├── config.rs         Qwen3Config presets: 0.6b, 4b, 8b, 14b
│   ├── network.rs        Qwen3Model network graph
│   ├── weights.rs        Safetensors weight loader
│   ├── tensor.rs         Tensor / DType definitions
│   ├── device_tensor.rs  RAII device tensor wrappers
│   ├── scratch_arena.rs  Per-layer scratch buffer pool
│   ├── quantize.rs       Quantization config
│   └── parallel.rs       TP/PP sharding config
├── ops/
│   ├── ascend.rs         Ascend NPU compute ops (aclnn)
│   └── ascend_comm.rs    HCCL collectives for distributed
├── scheduler/mod.rs      Tokenizer, request/response types, chat template
├── server/mod.rs         Axum HTTP server: /v1/completions, /v1/chat, /health, /v1/models
└── distributed/          DistributedConfig, process group init
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
| **Tests & Examples** | Require `TASK_DEVICE` env var — fail immediately via `expect` | `env::var("TASK_DEVICE").expect("TASK_DEVICE")` |
| **Entrypoint** (`main.rs`) | Full resolution chain with logging, graceful error if unset | `--device-id` > `LOCAL_RANK` > `TASK_DEVICE` > `ASCEND_DEVICE_ID` > error |

Tests never silently default to device 0. If `TASK_DEVICE` is missing, the test panics
with a stack trace pointing to the exact `expect()` call.

The canonical launcher is `task-submit --device auto`, which sets `TASK_DEVICE`.

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
