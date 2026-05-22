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

## Running One-Shot Generation

No HTTP server, just generate and print. The `--backend` default is
`"ascend"` when built with `--features ascend`, `"stub"` otherwise.

```bash
# Stub backend (CPU, no NPU needed)
# First, configure your model weights path:
#   cp scripts/env.example scripts/env
#   # Edit scripts/env and set MODEL_DIR to your weights directory
#
# Stub backend (CPU, no NPU needed):
cargo run -- \
  --weights /path/to/Qwen3-14B-weights \
  --prompt 'Huawei is' \
  --max-new-tokens 5 \
  --backend stub

# Ascend NPU (via task-submit with auto device detection)
# --backend ascend is the default when --features ascend is used:
# Preferred: use the wrapper script (reads MODEL_DIR from scripts/env)
scripts/run-one-shot --prompt 'Huawei is' --max-new-tokens 5

# Or directly via task-submit:
task-submit --device auto --run \
  "cd /path/to/serving_agent/rust_llm_server && \
   cargo run --features ascend -- \
     --weights /path/to/Qwen3-14B-weights \
     --prompt 'Huawei is' \
     --max-new-tokens 5"
```

Device auto-detection chain: `--device-id` > `TASK_DEVICE` (set by `task-submit`) > `ASCEND_DEVICE_ID` > 0.

## Running the HTTP Server

```bash
task-submit --device auto --run \
  "cd /path/to/serving_agent/rust_llm_server && \
   cargo run --features ascend -- \
     --weights /path/to/Qwen3-14B-weights"
```

Endpoints: `GET /health`, `POST /v1/completions`, `POST /v1/chat/completions`, `GET /v1/models`.
