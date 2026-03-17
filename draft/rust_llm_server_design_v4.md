# Rust LLM Inference Server — Design & Implementation Plan (v4)

> **Target:** 1–2 machines, 8–16 GPU/NPU cards, high-performance inference for agent/coding workloads
> **Philosophy:** Rust for orchestration and scheduling. Reuse CUDA/NPU kernels via FFI.
> **New in v4:** Detailed weight loading, memory profiling, CUDA graph capture, incremental detokenization, metrics/observability, error handling, and concrete test plans for every phase.

---

## Table of Contents

1. [Critique of v3 and What v4 Fixes](#1-critique-of-v3-and-what-v4-fixes)
2. [Configuration Space](#2-configuration-space)
3. [Generator Design](#3-generator-design)
4. [Architecture Overview](#4-architecture-overview)
5. [Hardware Abstraction Layer](#5-hardware-abstraction-layer)
6. [Model Config Detection and Weight Loading](#6-model-config-detection-and-weight-loading)
7. [Memory Profiling and KV Cache Allocation](#7-memory-profiling-and-kv-cache-allocation)
8. [Scheduler Design](#8-scheduler-design)
9. [KV Cache: Radix-Tree Prefix Cache](#9-kv-cache-radix-tree-prefix-cache)
10. [CUDA/ACL Graph Capture](#10-cudaacl-graph-capture)
11. [Incremental Detokenization](#11-incremental-detokenization)
12. [Pipeline Parallelism](#12-pipeline-parallelism)
13. [Expert Parallelism (MoE)](#13-expert-parallelism-moe)
14. [Disaggregated Serving](#14-disaggregated-serving)
15. [Quantization Schemes](#15-quantization-schemes)
16. [Kernel Reuse Map](#16-kernel-reuse-map)
17. [Metrics and Observability](#17-metrics-and-observability)
18. [Error Handling and Fault Tolerance](#18-error-handling-and-fault-tolerance)
19. [Key Rust Crates](#19-key-rust-crates)
20. [Implementation Phases with Test Plans](#20-implementation-phases-with-test-plans)
21. [Directory Layout](#21-directory-layout)
22. [Performance Targets](#22-performance-targets)

---

## 1. Critique of v3 and What v4 Fixes

### Critique

v3 introduced the right *ideas* (HAL, overlap scheduling, radix-tree cache, EP) but left critical implementation details unspecified. A team reading v3 would still not know *how* to build the system. Specific problems:

| # | v3 Problem | Why It Matters |
|---|---|---|
| 1 | **No weight loading design.** The entire flow from safetensors on disk to sharded GPU tensors is undescribed. | Weight loading is the first thing that runs. TP sharding rules differ by layer type (dim-0 for QKV, dim-1 for output projections). Getting this wrong corrupts every forward pass. |
| 2 | **No memory profiling.** v3 says "kv_cache_gb: 60" in the config but doesn't explain how the runtime decides the actual block count. | You can't hardcode KV cache size — it depends on model weight memory consumption, CUDA graph memory, and per-rank memory variance. The runtime must *measure* remaining memory after model load and warmup. |
| 3 | **No warmup phase.** v3 jumps from "load model" to "serve requests." | Warmup is mandatory to (a) profile peak transient memory, (b) pre-compile/JIT any lazy kernels, (c) populate graph caches. Without it, the first real request OOMs or has 10× latency. |
| 4 | **CUDA graph capture is vague.** "Decode steps are captured as CUDA graphs" — but which batch sizes? How is padding handled? What about graph pool reuse? | Graph capture consumes GPU memory. Capturing too many sizes wastes memory; too few means falling back to eager mode. Pool reuse is critical — without it, each graph allocates its own workspace. |
| 5 | **No incremental detokenization.** v3 mentions SSE streaming but not how tokens become characters. | Naïve `decode([token_id])` produces garbage for multi-byte UTF-8 characters and BPE fragments. Streaming requires a state machine that tracks partial characters across tokens. |
| 6 | **HAL traits are too abstract.** `DeviceBuffer` as a trait object adds vtable overhead on every tiny operation (index, slice). | For inner-loop operations (tensor indexing during attention metadata construction), trait objects are wrong. HAL should be at the *kernel dispatch* and *comm* level, not at the *buffer* level. Buffers should be a concrete enum. |
| 7 | **No model config detection.** How does the server know `num_layers`, `head_dim`, `num_kv_heads`? | HuggingFace model configs have inconsistent naming (`num_key_value_heads` vs `num_attention_heads`, `head_dim` vs computed from `hidden_size`). MoE fields vary between Mixtral and DeepSeek. This must be handled explicitly. |
| 8 | **No metrics or observability.** | You can't optimize what you can't measure. TTFT, TPOT, ITL, cache hit rate, and queue depth must be tracked from day one — they're how you know if each phase's checkpoint passes. |
| 9 | **No error handling design.** What happens on OOM? NCCL timeout? Malformed request? | Silent failures in a multi-GPU system are catastrophic. The design must specify recovery paths: OOM → preempt/swap, NCCL timeout → retry or abort batch, malformed request → 400 without crashing the engine. |
| 10 | **No test plans.** Phase checkpoints say "Llama-3.1-8B serves end-to-end" but not *how to verify*. | Without concrete test procedures (input, expected output, pass/fail criteria, benchmarks), checkpoints are aspirational. |
| 11 | **Radix tree concurrency is unspecified.** `Arc<RwLock<RadixNode>>` per node is a recipe for lock contention under high concurrency. | The scheduler accesses the prefix cache on the hot path. Lock granularity and eviction policy must be designed carefully. |
| 12 | **Ascend NPU specifics are generic.** v3 says "CANN ops" without naming them. | Ascend's attention kernels (`dlinfer` ops), KV cache block shapes, and HCCL MoE comm type selection (MC2 vs AllToAll depending on SOC version) are all hardware-specific details that affect correctness. |
| 13 | **Disaggregated serving is hand-wavy.** "KV transfer via IPC or RDMA" without a protocol, sizing, or latency analysis. | KV transfer for a 131K-token request with 80 layers is ~40 GB. The transfer mechanism determines whether disagg mode is practical. |

### What v4 Adds

v4 retains all v3 ideas and adds:

- **§6**: Concrete weight loading with TP sharding rules per layer type
- **§7**: Memory profiling formula, warmup procedure, per-rank memory balancing
- **§8**: Refined scheduler with explicit state machine and preemption protocol
- **§9**: Radix tree with epoch-based reclamation (no per-node RwLock)
- **§10**: CUDA graph capture with dynamic batch sizes, pool reuse, padding protocol
- **§11**: Incremental detokenizer with surrogate-pair state machine
- **§17**: Full metrics schema (TTFT, TPOT, ITL, cache hit, queue depth)
- **§18**: Error handling for OOM, NCCL failure, request timeout, malformed input
- **§20**: Detailed test plans for every phase

---

## 2. Configuration Space

*(Unchanged from v3. See v3 §2 for parallelism modes, quantization, hardware targets, and example configs.)*

One addition — the config now includes warmup and profiling settings:

```yaml
runtime:
  gpu_memory_utilization: 0.90    # fraction of device memory available for KV cache
  memory_balance_threshold_gb: 2  # max imbalance across TP ranks before error
  warmup_batch_tokens: 16384      # tokens to use in warmup pass
  cuda_graph_max_bs: 0            # 0 = auto-detect from free memory
  cuda_graph_batch_sizes: []      # explicit override, e.g. [1, 2, 4, 8, 16, 32]
  enforce_eager: false             # disable graph capture
```

---

## 3. Generator Design

*(Unchanged from v3 §3. The generator now also emits a `warmup.rs` stub with the profiling procedure.)*

---

## 4. Architecture Overview

```
                              ┌──────────────────────────┐
                              │   llm-gen (generator)    │
                              │  config.yaml → project   │
                              └────────────┬─────────────┘
                                           │ generates
┌──────────────────────────────────────────▼──────────────────────────────────────┐
│                              Specialized Binary                                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         Startup Sequence                                │    │
│  │  1. Parse CLI args (--model, --tp-rank, --pp-rank, --role)             │    │
│  │  2. Init HAL (CUDA context or ACL init)                                │    │
│  │  3. Init comm (NCCL/HCCL)                                              │    │
│  │  4. Read HF model config → ModelConfig                                  │    │
│  │  5. Load + shard weights                                                │    │
│  │  6. Warmup: forward pass with dummy batch → measure peak memory         │    │
│  │  7. Allocate KV cache blocks from remaining memory                      │    │
│  │  8. Capture CUDA/ACL graphs for decode batch sizes                      │    │
│  │  9. Start HTTP server → accept requests                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────────┐     │
│  │  HTTP API (axum)  │→│  Overlap Scheduler   │→│  Executor (Simple/PP)  │     │
│  │  /v1/chat/compl.  │  │  Token-budget        │  │  HAL kernel dispatch   │     │
│  │  SSE streaming    │  │  Chunked prefill     │  │  TP AllReduce          │     │
│  └──────────────────┘  │  Preemption           │  │  Graph replay          │     │
│                         └──────────┬────────────┘  └──────────┬─────────────┘     │
│                                    │                           │                   │
│  ┌─────────────────────────────────▼───────────────────────────▼───────────┐     │
│  │                     Radix-Tree KV Manager                               │     │
│  │  Prefix match · Block alloc · LRU evict · CPU swap                      │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐     │
│  │              Metrics Collector (prometheus + tracing)                    │     │
│  │  TTFT · TPOT · ITL · cache_hit_rate · queue_depth · gpu_util            │     │
│  └─────────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Hardware Abstraction Layer

### v4 Revision: Concrete Buffer Enum, Trait-Based Dispatch Only at Kernel Level

v3's mistake was making `DeviceBuffer` a trait object. Buffer operations (pointer arithmetic, slicing) are too hot-path for virtual dispatch. Instead:

```rust
/// Concrete enum — no vtable. The generator picks one variant and the compiler
/// eliminates the other via const propagation on HARDWARE.
pub enum DeviceBuffer {
    Cuda(CudaBuffer),    // wraps CUdeviceptr + len
    Ascend(AscendBuffer), // wraps aclDataBuffer + len
}

impl DeviceBuffer {
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Cuda(b) => b.ptr as *const u8,
            Self::Ascend(b) => b.ptr as *const u8,
        }
    }

    #[inline(always)]
    pub fn len_bytes(&self) -> usize {
        match self {
            Self::Cuda(b) => b.len,
            Self::Ascend(b) => b.len,
        }
    }
}
```

Since `HARDWARE` is a compile-time constant, the match arms for the unused variant are dead-code-eliminated. This gives us zero-cost hardware abstraction for buffers.

### Kernel Dispatch: Trait Object (Acceptable)

Kernel calls take microseconds+. One vtable lookup is immeasurable:

```rust
pub trait KernelDispatch: Send + Sync {
    fn attention_prefill(&self, params: &AttentionPrefillParams, stream: &Stream);
    fn attention_decode_paged(&self, params: &AttentionDecodeParams, stream: &Stream);
    fn gemm(&self, params: &GemmParams, stream: &Stream);
    fn rms_norm(&self, params: &RmsNormParams, stream: &Stream);
    fn rotary_emb(&self, params: &RotaryParams, stream: &Stream);
    fn silu_mul(&self, params: &ActivationParams, stream: &Stream);
    fn top_p_sampling(&self, params: &SamplingParams, stream: &Stream);
    fn fill_kv_cache(&self, params: &FillKvParams, stream: &Stream);
}
```

### Comm Backend: Trait Object

```rust
pub trait CommBackend: Send + Sync {
    fn all_reduce_sum(&self, buf: &mut DeviceBuffer, count: usize, dtype: Dtype, stream: &Stream);
    fn send(&self, buf: &DeviceBuffer, count: usize, dtype: Dtype, dest: usize, stream: &Stream);
    fn recv(&self, buf: &mut DeviceBuffer, count: usize, dtype: Dtype, src: usize, stream: &Stream);
    fn all_to_all(&self, send: &DeviceBuffer, recv: &mut DeviceBuffer,
                  send_counts: &[usize], recv_counts: &[usize], stream: &Stream);
}
```

### Stream

```rust
pub enum Stream {
    Cuda(CudaStream),    // wraps cudaStream_t
    Ascend(AclStream),   // wraps aclrtStream
}

impl Stream {
    pub fn synchronize(&self);
    pub fn record_event(&self) -> Event;
    pub fn wait_event(&self, event: &Event);
}
```

### Ascend-Specific: KV Cache Block Shape

On Ascend 910, the KV cache block layout differs from CUDA:

```rust
// CUDA: [num_blocks, 2, num_kv_heads, block_size, head_dim]
// Ascend 910: [num_blocks, 2, block_size, num_kv_heads, head_dim]
//   (block_size and num_kv_heads are transposed)

pub const fn kv_block_shape() -> [usize; 5] {
    match HARDWARE {
        Hardware::Cuda => [NUM_BLOCKS, 2, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM],
        Hardware::Ascend => [NUM_BLOCKS, 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM],
    }
}
```

This is emitted by the generator based on `hardware` config.

### Ascend-Specific: MoE Communication Type

Ascend 910 A2 and A3 have different optimal MoE communication patterns:

```rust
pub enum MoeCommType {
    AllGather,   // EP ≤ 1 or A2 with large token count
    AllToAll,    // A3 with large token count
    Mc2,         // Multi-Cast Collective — A2/A3 with small token count
}

pub fn select_moe_comm_type(
    max_tokens: usize,
    dp_size: usize,
    tp_size: usize,
    ep_size: usize,
) -> MoeCommType {
    if ep_size <= 1 { return MoeCommType::AllGather; }

    let mc2_capacity = 4096; // empirical threshold
    match ASCEND_SOC {
        AscendSoc::A2 => {
            if max_tokens <= mc2_capacity && dp_size * tp_size >= 16 {
                MoeCommType::Mc2
            } else {
                MoeCommType::AllGather
            }
        }
        AscendSoc::A3 => {
            if max_tokens <= mc2_capacity {
                MoeCommType::Mc2
            } else {
                MoeCommType::AllToAll
            }
        }
    }
}
```

---

## 6. Model Config Detection and Weight Loading

### 6.1 Reading HuggingFace Model Config

HuggingFace models store their architecture in `config.json`. The naming is inconsistent across model families. Our `ModelConfig` normalizes this:

```rust
#[derive(Deserialize)]
pub struct HfConfig {
    // Common
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: Option<f64>,
    pub max_position_embeddings: usize,

    // GQA — may be absent (= num_attention_heads)
    pub num_key_value_heads: Option<usize>,

    // Head dim — may be absent (= hidden_size / num_attention_heads)
    pub head_dim: Option<usize>,

    // RoPE — may be top-level or nested inside rope_scaling
    pub rope_theta: Option<f64>,
    pub rope_scaling: Option<RopeScaling>,

    // MoE — field names differ between models
    pub num_local_experts: Option<usize>,   // Mixtral style
    pub num_experts: Option<usize>,         // DeepSeek style
    pub num_experts_per_tok: Option<usize>, // Mixtral
    pub moe_intermediate_size: Option<usize>,

    // Multimodal: check text_config first
    pub text_config: Option<Box<HfConfig>>,

    // Tie embedding weights
    pub tie_word_embeddings: Option<bool>,
}

pub struct ModelConfig {
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub rms_norm_eps: f64,
    pub num_experts: usize,        // 0 for dense models
    pub num_experts_per_tok: usize, // 0 for dense models
    pub moe_intermediate_size: usize,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
}

impl ModelConfig {
    pub fn from_hf(hf: HfConfig) -> Self {
        // If multimodal, unwrap text_config
        let cfg = if let Some(text) = hf.text_config { *text } else { hf };

        let num_kv_heads = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads);
        let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rope_theta = cfg.rope_theta
            .or_else(|| cfg.rope_scaling.as_ref()?.rope_theta)
            .unwrap_or(10000.0);
        let num_experts = cfg.num_local_experts.or(cfg.num_experts).unwrap_or(0);

        ModelConfig {
            num_layers: cfg.num_hidden_layers,
            num_q_heads: cfg.num_attention_heads,
            num_kv_heads,
            head_dim,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            rope_theta,
            rope_scaling: cfg.rope_scaling,
            rms_norm_eps: cfg.rms_norm_eps.unwrap_or(1e-6),
            num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok.unwrap_or(0),
            moe_intermediate_size: cfg.moe_intermediate_size.unwrap_or(0),
            tie_word_embeddings: cfg.tie_word_embeddings.unwrap_or(false),
            max_position_embeddings: cfg.max_position_embeddings,
        }
    }
}
```

### 6.2 Weight Loading

#### Safetensors Discovery

```rust
fn discover_weight_files(model_dir: &Path) -> Vec<PathBuf> {
    // Priority: single file → sharded index → pytorch fallback
    let single = model_dir.join("model.safetensors");
    if single.exists() { return vec![single]; }

    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let idx: SafetensorsIndex = serde_json::from_reader(File::open(&index)?)?;
        return idx.weight_map.values()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|f| model_dir.join(f))
            .collect();
    }

    // Glob fallback
    glob(model_dir.join("*.safetensors"))
}
```

#### TP Sharding Rules

Different tensor types shard on different dimensions:

```rust
#[derive(Clone, Copy)]
enum ShardDim {
    Dim0,  // Split along rows — each rank gets rows [rank * chunk .. (rank+1) * chunk]
    Dim1,  // Split along columns
    None,  // Replicated on all ranks (norms, biases, embed if not vocab-parallel)
}

/// Determines shard dimension for a given parameter name.
fn shard_rule(name: &str, tp_size: usize, num_kv_heads: usize) -> ShardDim {
    if tp_size == 1 { return ShardDim::None; }

    // QKV projections: split along output dim (dim 0)
    if name.contains("q_proj.weight")
        || name.contains("k_proj.weight")
        || name.contains("v_proj.weight")
        || name.contains("gate_proj.weight")
        || name.contains("up_proj.weight")
    {
        return ShardDim::Dim0;
    }

    // Output / down projections: split along input dim (dim 1)
    if name.contains("o_proj.weight")
        || name.contains("down_proj.weight")
    {
        return ShardDim::Dim1;
    }

    // Embedding: vocab-parallel split on dim 0
    if name.contains("embed_tokens.weight") || name.contains("lm_head.weight") {
        return ShardDim::Dim0;
    }

    // MoE expert weights: shard per expert, then shard expert's weight
    if name.contains("experts.") {
        // Parse expert index from name, assign to EP rank
        // Then shard the expert's weight by TP like dense layers
        return expert_shard_rule(name, tp_size);
    }

    // Norms, biases: replicated
    ShardDim::None
}
```

#### Special: KV Head Sharding When num_kv_heads < TP

When `num_kv_heads < tp_size` (e.g., GQA with 8 KV heads on TP=16), we can't simply chunk:

```rust
fn shard_kv_heads(
    tensor: &Tensor,       // shape: [num_kv_heads * head_dim, hidden_size]
    tp_rank: usize,
    tp_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    if num_kv_heads >= tp_size {
        // Simple chunk along dim 0
        let chunk = num_kv_heads / tp_size;
        let start = tp_rank * chunk * head_dim;
        let end = start + chunk * head_dim;
        tensor.slice(0, start, end)
    } else {
        // Replicate heads: each rank gets head_index = tp_rank % num_kv_heads
        let head_idx = tp_rank % num_kv_heads;
        let start = head_idx * head_dim;
        let end = start + head_dim;
        tensor.slice(0, start, end)
    }
}
```

#### Fused QKV Loading

Many models store Q, K, V as separate tensors. We fuse them at load time for a single GEMM:

```rust
/// Buffer Q, K, V projections, then fuse into a single [qkv_dim, hidden_dim] tensor.
struct QkvFuser {
    q: Option<Tensor>,
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl QkvFuser {
    fn try_fuse(&mut self) -> Option<Tensor> {
        if self.q.is_some() && self.k.is_some() && self.v.is_some() {
            let fused = Tensor::cat(&[
                self.q.take().unwrap(),
                self.k.take().unwrap(),
                self.v.take().unwrap(),
            ], /*dim=*/ 0);
            Some(fused)
        } else {
            None
        }
    }
}
```

#### MoE Expert Stacking

MoE models store each expert separately. We stack them into a single tensor for batched GEMM:

```rust
/// Buffer experts for a layer, then stack into [num_experts, out_dim, in_dim].
struct ExpertStacker {
    experts: Vec<Option<Tensor>>,
    total: usize,
}

impl ExpertStacker {
    fn insert(&mut self, expert_idx: usize, tensor: Tensor) {
        self.experts[expert_idx] = Some(tensor);
    }

    fn try_stack(&self) -> Option<Tensor> {
        if self.experts.iter().all(|e| e.is_some()) {
            let tensors: Vec<&Tensor> = self.experts.iter().map(|e| e.as_ref().unwrap()).collect();
            Some(Tensor::stack(&tensors, 0))
        } else {
            None
        }
    }
}
```

#### Weight Loading Pipeline

```rust
pub fn load_weights(
    model_dir: &Path,
    model_config: &ModelConfig,
    tp_rank: usize,
    tp_size: usize,
    pp_rank: usize,
    pp_size: usize,
    device: &Device,
) -> ModelWeights {
    let files = discover_weight_files(model_dir);
    let my_layers = layer_range(model_config.num_layers, pp_rank, pp_size);
    let mut weights = ModelWeights::new(model_config, my_layers.len());
    let mut qkv_fusers: HashMap<usize, QkvFuser> = HashMap::new();
    let mut expert_stackers: HashMap<(usize, String), ExpertStacker> = HashMap::new();

    for file in &files {
        let st = SafeTensors::open(file)?;
        for (name, tensor_view) in st.tensors() {
            // 1. Skip layers not owned by this PP rank
            if let Some(layer_idx) = parse_layer_index(&name) {
                if !my_layers.contains(&layer_idx) { continue; }
            }

            // 2. Determine shard rule and shard
            let rule = shard_rule(&name, tp_size, model_config.num_kv_heads);
            let tensor = match rule {
                ShardDim::Dim0 => shard_dim0(&tensor_view, tp_rank, tp_size),
                ShardDim::Dim1 => shard_dim1(&tensor_view, tp_rank, tp_size),
                ShardDim::None => tensor_view.to_tensor(),
            };

            // 3. Handle KV head special case
            let tensor = if name.contains("k_proj") || name.contains("v_proj") {
                shard_kv_heads(&tensor, tp_rank, tp_size,
                               model_config.num_kv_heads, model_config.head_dim)
            } else {
                tensor
            };

            // 4. Convert dtype if quantized (BF16 → FP8, INT8, INT4)
            let tensor = maybe_quantize(tensor, &name);

            // 5. Copy to device
            let tensor = tensor.to_device(device);

            // 6. Store — with QKV fusion and expert stacking
            weights.insert(&name, tensor, &mut qkv_fusers, &mut expert_stackers);
        }
    }

    weights
}
```

---

## 7. Memory Profiling and KV Cache Allocation

### 7.1 The Problem

You cannot know at config time how much memory is available for KV cache. The model's weight memory depends on quantization, TP sharding, and layer assignment (PP). CUDA graph memory depends on batch sizes. Transient memory (activations during warmup) fluctuates.

### 7.2 The Procedure

```
┌─────────────────────────────────────────┐
│ 1. Record free memory before model load │  free_before = device_free_memory()
│ 2. Load model weights                   │
│ 3. Record free memory after model load  │  free_after = device_free_memory()
│    model_memory = free_before - free_after
│ 4. Clear caches, reset peak stats       │
│ 5. Warmup: forward pass with max tokens │  → records peak transient alloc
│ 6. Read peak memory stats               │  peak_mem = device_peak_allocated()
│    transient_mem = peak_mem - current_allocated()
│ 7. Clear caches again                   │
│ 8. Compute available memory for KV:     │
│    available = total * utilization       │
│                - model_memory            │
│                - transient_mem           │
│                - graph_estimate          │
│ 9. Compute block count:                 │
│    block_bytes = 2 * num_layers_this_rank * block_size
│                  * num_kv_heads_per_rank * head_dim * kv_dtype_size
│    num_blocks = available / block_bytes  │
│10. Balance across TP ranks (all-reduce min) │
└─────────────────────────────────────────┘
```

### 7.3 Implementation

```rust
pub fn allocate_kv_cache(
    device: &Device,
    model_config: &ModelConfig,
    runtime_config: &RuntimeConfig,
    comm: &dyn CommBackend,
) -> BlockPool {
    // Step 1: measure current state
    let (free, total) = device.mem_info();
    let current_alloc = device.current_allocated();
    let peak_alloc = device.peak_allocated();

    // Step 2: transient memory = peak during warmup minus steady-state
    let transient = peak_alloc - current_alloc;

    // Step 3: estimate graph memory (heuristic: ~200 MB per captured batch size)
    let num_graph_sizes = if runtime_config.enforce_eager {
        0
    } else if runtime_config.cuda_graph_batch_sizes.is_empty() {
        estimated_graph_count(free)  // see §10
    } else {
        runtime_config.cuda_graph_batch_sizes.len()
    };
    let graph_estimate = num_graph_sizes * 200 * 1024 * 1024; // rough

    // Step 4: available memory
    let target = (total as f64 * runtime_config.gpu_memory_utilization) as usize;
    let used = total - free;
    let available = target.saturating_sub(used + transient + graph_estimate);

    // Step 5: block size in bytes
    let layers_this_rank = model_config.num_layers / PP_SIZE;
    let kv_heads_per_rank = model_config.num_kv_heads / TP_SIZE;
    let block_bytes = 2  // K + V
        * layers_this_rank
        * BLOCK_SIZE
        * kv_heads_per_rank
        * model_config.head_dim
        * kv_dtype_size();  // 2 for BF16, 1 for FP8/INT8

    let mut num_blocks = available / block_bytes;

    // Step 6: balance across TP ranks — take minimum
    num_blocks = sync_min_across_ranks(num_blocks, comm);

    // Step 7: sanity check
    let threshold = runtime_config.memory_balance_threshold_gb * 1024 * 1024 * 1024;
    let max_blocks = sync_max_across_ranks(num_blocks, comm);
    if (max_blocks - num_blocks) * block_bytes > threshold {
        panic!("Memory imbalance across TP ranks exceeds {}GB", runtime_config.memory_balance_threshold_gb);
    }

    tracing::info!(num_blocks, block_bytes, available, "KV cache allocated");
    BlockPool::new(device, num_blocks, layers_this_rank, kv_heads_per_rank, model_config.head_dim)
}
```

### 7.4 Warmup Procedure

```rust
pub fn warmup(
    executor: &dyn Executor,
    model_config: &ModelConfig,
    runtime_config: &RuntimeConfig,
    device: &Device,
) {
    device.empty_cache();
    device.reset_peak_memory_stats();

    // Create a dummy batch at maximum token budget
    let max_tokens = runtime_config.warmup_batch_tokens;
    let max_seq_len = model_config.max_position_embeddings.min(MAX_SEQ_LEN);
    let num_seqs = (max_tokens / max_seq_len).max(1).min(MAX_DECODE_SEQS);
    let seq_len = max_tokens / num_seqs;

    let dummy_input = ForwardInput::dummy_prefill(num_seqs, seq_len);
    executor.forward_sync(&dummy_input);  // blocking — no overlap

    device.synchronize();
    // Peak stats are now recorded and will be used by allocate_kv_cache
    device.empty_cache();

    tracing::info!(
        peak_mb = device.peak_allocated() / (1024 * 1024),
        current_mb = device.current_allocated() / (1024 * 1024),
        "Warmup complete"
    );
}
```

---

## 8. Scheduler Design

### 8.1 Sequence State Machine

```
                     ┌───────────┐
         new request │  WAITING  │ queued for prefill
                     └─────┬─────┘
                           │ scheduler picks, blocks allocated
                     ┌─────▼─────┐
              ┌──────│  RUNNING  │──────┐
              │      └─────┬─────┘      │
        preempt (OOM)      │        finish (EOS/max_len/stop)
              │            │            │
      ┌───────▼──────┐    │    ┌───────▼──────┐
      │   SWAPPED    │    │    │   FINISHED   │
      └───────┬──────┘    │    └──────────────┘
              │            │
          resume           │ error (OOM unrecoverable, timeout)
              │      ┌─────▼─────┐
              └──────│  RUNNING  │
                     └───────────┘
```

```rust
pub enum SeqState {
    Waiting,
    Running { is_prefill: bool },
    Swapped,
    Finished(FinishReason),
}

pub enum FinishReason {
    Eos,
    StopToken(u32),
    StopString(String),
    MaxTokens,
    MaxModelLen,
    Timeout,
    Canceled,
    Error(String),
}
```

### 8.2 Overlap Scheduler

```rust
pub struct OverlapScheduler {
    waiting: VecDeque<Arc<Sequence>>,
    running: Vec<Arc<Sequence>>,
    swapped: Vec<Arc<Sequence>>,

    kv_manager: Arc<RadixKvManager>,
    token_budget: usize,       // = MAX_BATCH_TOKENS
    max_decode_seqs: usize,
    max_prefill_tokens: usize, // = MAX_PREFILL_TOKENS

    // Metrics (see §17)
    metrics: Arc<MetricsCollector>,
}
```

#### Schedule Step (CPU side)

```rust
fn schedule(&mut self) -> ScheduleResult {
    let mut budget = self.token_budget;
    let mut decode_seqs = Vec::new();
    let mut prefill_seqs = Vec::new();
    let mut preempt_seqs = Vec::new();

    // Phase 1: Decode — each running sequence needs 1 token
    for seq in &self.running {
        if budget == 0 { break; }
        // Check KV block availability: need to append 1 token to last block
        if !self.kv_manager.can_append(seq) {
            // Preempt: pick victim (longest sequence or lowest priority)
            if let Some(victim) = self.pick_preempt_victim() {
                self.swap_out(&victim);
                preempt_seqs.push(victim);
            } else {
                // Cannot preempt anyone — this sequence must wait
                continue;
            }
        }
        decode_seqs.push(Arc::clone(seq));
        budget -= 1;
    }

    // Phase 2: Prefill — fill remaining budget with chunked prefill
    //          Only if no preemptions occurred (to avoid thrashing)
    if preempt_seqs.is_empty() {
        let prefill_budget = budget.min(self.max_prefill_tokens);
        let mut prefill_used = 0;

        while let Some(seq) = self.waiting.front() {
            // Token layout: how many tokens are cached, how many to compute?
            let layout = self.kv_manager.get_token_layout(seq);
            let tokens_this_step = (seq.prompt_len() - layout.computed_tokens)
                .min(prefill_budget - prefill_used);

            if tokens_this_step == 0 { break; }

            // Allocate blocks for the new tokens
            let blocks_needed = (tokens_this_step + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if !self.kv_manager.can_allocate(blocks_needed) { break; }

            let seq = self.waiting.pop_front().unwrap();
            self.kv_manager.allocate_for(seq.id(), blocks_needed);
            prefill_seqs.push((seq, layout, tokens_this_step));
            prefill_used += tokens_this_step;
        }
    }

    ScheduleResult { decode_seqs, prefill_seqs, preempt_seqs }
}
```

#### Overlap Loop

```rust
pub async fn run(mut self, executor: Arc<dyn Executor>) {
    // Channel-based pipelining
    let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<ForwardInput>(1);
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<ForwardOutput>(1);

    // GPU task
    let exec = executor.clone();
    tokio::spawn(async move {
        while let Some(input) = batch_rx.recv().await {
            let output = exec.forward(input).await;
            let _ = result_tx.send(output).await;
        }
    });

    // CPU task: schedule, process results, feed batches
    let mut prev_result: Option<ForwardOutput> = None;
    loop {
        // 1. Process previous step's output (while GPU may still be starting next step)
        if let Some(result) = prev_result.take() {
            self.process_results(&result).await;
        }

        // 2. Accept new requests from HTTP layer
        self.drain_request_queue();

        // 3. Schedule next batch
        let schedule = self.schedule();
        if schedule.is_empty() {
            // Nothing to do — wait for new requests or GPU result
            tokio::select! {
                result = result_rx.recv() => {
                    prev_result = result;
                }
                _ = self.new_request_signal.notified() => {}
            }
            continue;
        }

        let input = self.build_forward_input(&schedule);

        // 4. Send batch to GPU task (may block if GPU is busy)
        batch_tx.send(input).await.unwrap();

        // 5. Wait for previous batch result
        if let Some(result) = result_rx.recv().await {
            prev_result = Some(result);
        }
    }
}
```

#### Process Results

```rust
async fn process_results(&mut self, output: &ForwardOutput) {
    for (seq_id, token_id) in &output.sampled_tokens {
        let seq = self.find_running_mut(*seq_id);

        // Record timing
        if seq.generated_tokens() == 0 {
            seq.mark_first_token_time();  // for TTFT
        }
        seq.mark_latest_token_time();  // for TPOT/ITL

        // Append token
        seq.append_token(*token_id);

        // Insert newly-computed KV blocks into prefix cache
        self.kv_manager.insert_prefix(seq.all_tokens(), seq.block_ids());

        // Check finish conditions
        if seq.is_finished() {
            seq.mark_finish_time();
            self.metrics.record_request_complete(seq);
            self.running.retain(|s| s.id() != *seq_id);
            self.notify_request_done(*seq_id);
        }
    }
}
```

---

## 9. KV Cache: Radix-Tree Prefix Cache

### 9.1 v4 Revision: Epoch-Based Reclamation, No Per-Node RwLock

v3 put `Arc<RwLock<...>>` on every radix tree node — under high concurrency with hundreds of sequences, this causes lock contention on the hot path.

v4 uses **epoch-based reclamation** (similar to `crossbeam-epoch`): readers enter an epoch (cost: one atomic increment), traverse the tree without locks, and writers (insert, evict) take a single global write lock that blocks only other writers.

This works because:
- The scheduler is single-threaded → only one writer at a time
- Multiple reader contexts (metrics, debug inspection) are rare
- Eviction happens in the scheduler thread, not concurrently

```rust
pub struct RadixKvManager {
    /// The tree root. Children of root are the first-block entries.
    root: RadixNode,
    /// Block pool for GPU memory
    gpu_pool: BlockPool,
    /// Block pool for CPU swap
    cpu_pool: BlockPool,
    /// Eviction candidates sorted by last_used
    eviction_queue: BinaryHeap<Reverse<(u64, u32)>>,  // (timestamp, block_id)
    /// Monotonic clock for LRU
    clock: AtomicU64,
}

struct RadixNode {
    /// Token content of this block (BLOCK_SIZE tokens, or fewer for the last block)
    tokens: SmallVec<[u32; BLOCK_SIZE]>,
    /// Physical block ID on the device (or INVALID for root)
    block_id: u32,
    /// Children keyed on first token of next block — sorted vec for cache locality
    children: Vec<(u32, Box<RadixNode>)>,
    /// Active reference count (how many live sequences use this block)
    ref_count: u32,
    /// Last access timestamp for LRU eviction
    last_used: u64,
}
```

#### Sorted Vec for Children

Using `Vec<(u32, Box<RadixNode>)>` sorted by first token instead of `HashMap`:
- Typical node has 1–4 children (branching factor is low for real workloads)
- Linear scan on 1–4 elements is faster than HashMap's hashing + allocation
- Cache-friendly: all children pointers are contiguous

```rust
impl RadixNode {
    fn find_child(&self, first_token: u32) -> Option<&RadixNode> {
        self.children.iter()
            .find(|(tok, _)| *tok == first_token)
            .map(|(_, node)| node.as_ref())
    }

    fn find_child_mut(&mut self, first_token: u32) -> Option<&mut RadixNode> {
        self.children.iter_mut()
            .find(|(tok, _)| *tok == first_token)
            .map(|(_, node)| node.as_mut())
    }

    fn insert_child(&mut self, first_token: u32, node: Box<RadixNode>) {
        let pos = self.children.partition_point(|(t, _)| *t < first_token);
        self.children.insert(pos, (first_token, node));
    }
}
```

#### Core Operations

```rust
impl RadixKvManager {
    /// Find longest matching prefix in the tree.
    /// Returns (num_cached_tokens, block_ids).
    /// Increments ref_count on matched nodes.
    pub fn find_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<u32>) {
        let now = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut node = &mut self.root;
        let mut cached = 0;
        let mut block_ids = Vec::new();

        for chunk in tokens.chunks(BLOCK_SIZE) {
            if chunk.len() < BLOCK_SIZE { break; } // partial block: can't cache
            let first = chunk[0];
            match node.find_child_mut(first) {
                Some(child) if child.tokens.as_slice() == chunk => {
                    child.ref_count += 1;
                    child.last_used = now;
                    block_ids.push(child.block_id);
                    cached += BLOCK_SIZE;
                    node = child;
                }
                _ => break,
            }
        }
        (cached, block_ids)
    }

    /// Insert newly computed blocks into the tree.
    pub fn insert_prefix(&mut self, tokens: &[u32], block_ids: &[u32]) {
        let now = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut node = &mut self.root;

        for (chunk, &block_id) in tokens.chunks(BLOCK_SIZE).zip(block_ids) {
            if chunk.len() < BLOCK_SIZE { break; }
            let first = chunk[0];
            if node.find_child(first).is_some() {
                // Already cached — just traverse
                node = node.find_child_mut(first).unwrap();
                node.last_used = now;
            } else {
                // Insert new node
                let new_node = Box::new(RadixNode {
                    tokens: SmallVec::from_slice(chunk),
                    block_id,
                    children: Vec::new(),
                    ref_count: 0,
                    last_used: now,
                });
                node.insert_child(first, new_node);
                node = node.find_child_mut(first).unwrap();
            }
        }
    }

    /// Release prefix blocks (decrement ref_count).
    pub fn release(&mut self, block_ids: &[u32]) {
        // Walk tree to find nodes, decrement ref_count
        // Nodes with ref_count == 0 become eviction candidates
        self.release_recursive(&mut self.root, block_ids, 0);
    }

    /// Evict blocks until `target` blocks are freed.
    pub fn evict(&mut self, target: usize) -> usize {
        let mut freed = 0;
        while freed < target {
            // Find leaf node with ref_count == 0 and oldest last_used
            if let Some(leaf) = self.find_oldest_unreferenced_leaf() {
                self.gpu_pool.free(leaf.block_id);
                self.remove_leaf(leaf);
                freed += 1;
            } else {
                break; // no more evictable blocks
            }
        }
        freed
    }

    /// Token layout prediction (from Nano-vLLM-v1).
    pub fn get_token_layout(&mut self, seq: &Sequence) -> TokenLayout {
        let (cached, _) = self.find_prefix(seq.all_tokens());
        // Release immediately — we're just probing, not committing
        // (find_prefix incremented ref_counts; undo that)
        self.release(&self.find_prefix(seq.all_tokens()).1);

        TokenLayout {
            computed_tokens: cached,
            total_tokens: seq.prompt_len(),
            remaining: seq.prompt_len() - cached,
        }
    }
}
```

---

## 10. CUDA/ACL Graph Capture

### 10.1 Dynamic Batch Size Selection

Based on available free memory after KV cache allocation:

```rust
fn compute_graph_batch_sizes(
    config: &RuntimeConfig,
    free_memory: usize,
) -> Vec<usize> {
    if config.enforce_eager {
        return vec![];
    }

    // Explicit override
    if !config.cuda_graph_batch_sizes.is_empty() {
        return config.cuda_graph_batch_sizes.clone();
    }

    // Auto-detect based on free memory (after KV cache allocation)
    let free_gb = free_memory as f64 / (1 << 30) as f64;
    let max_bs = if free_gb > 80.0 {
        256  // H200-class
    } else if free_gb > 40.0 {
        160  // H100 80GB
    } else if free_gb > 16.0 {
        128
    } else {
        64
    };

    let max_bs = config.cuda_graph_max_bs
        .filter(|&bs| bs > 0)
        .unwrap_or(max_bs);

    // Small sizes for tail latency, then stride of 8
    let mut sizes = vec![1, 2, 4];
    let mut bs = 8;
    while bs <= max_bs {
        sizes.push(bs);
        bs += 8;
    }
    sizes
}
```

### 10.2 Graph Pool Reuse

Critical: capture graphs in **reverse order** (largest to smallest) and reuse the memory pool from the first captured graph:

```rust
pub fn capture_graphs(
    executor: &dyn Executor,
    batch_sizes: &[usize],
    device: &Device,
) -> HashMap<usize, CapturedGraph> {
    let mut graphs = HashMap::new();
    let mut pool: Option<GraphPool> = None;

    // Reverse order: largest first → allocates the most memory → pool is big enough for all
    for &bs in batch_sizes.iter().rev() {
        device.synchronize();

        // Warm up this batch size (required before capture)
        let dummy = ForwardInput::dummy_decode(bs);
        executor.forward_sync(&dummy);
        device.synchronize();

        // Capture
        let graph = device.begin_graph_capture(pool.as_ref());
        executor.forward_sync(&dummy);  // captured!
        let captured = device.end_graph_capture(graph);

        // Reuse pool from first (largest) graph
        if pool.is_none() {
            pool = Some(captured.pool());
        }

        graphs.insert(bs, captured);
        tracing::info!(batch_size = bs, "Captured CUDA graph");
    }

    graphs
}
```

### 10.3 Padding to Captured Batch Size

At inference time, pad the actual batch to the next captured size:

```rust
fn select_graph(&self, actual_bs: usize) -> Option<(&CapturedGraph, usize)> {
    // Find smallest captured size >= actual_bs
    let padded_bs = self.graph_batch_sizes.iter()
        .copied()
        .find(|&bs| bs >= actual_bs)?;

    let graph = self.graphs.get(&padded_bs)?;
    Some((graph, padded_bs))
}

fn forward_decode(&self, input: &ForwardInput) -> ForwardOutput {
    let actual_bs = input.decode_seqs.len();

    if let Some((graph, padded_bs)) = self.select_graph(actual_bs) {
        // Copy real inputs into graph's input buffer, pad remainder with dummy
        self.fill_graph_inputs(graph, input, padded_bs);
        graph.replay();
        // Slice output to actual batch size
        self.extract_graph_outputs(graph, actual_bs)
    } else {
        // Eager fallback for sizes larger than any captured graph
        self.forward_eager(input)
    }
}
```

### 10.4 Graph Capture on Ascend

Ascend uses `dlinfer`'s AscendGraphRunner. The main difference: batch sizes are quantized to "compatible sizes" (powers of 2 or Ascend-specific strides):

```rust
fn ascend_compatible_size(actual: usize) -> usize {
    // Ascend 910 prefers sizes aligned to 16 for DMA efficiency
    ((actual + 15) / 16) * 16
}
```

### 10.5 What Cannot Be Captured

NCCL/HCCL collective ops **cannot** be inside a graph. The capture boundary is around the per-GPU local computation; AllReduce/P2P are outside:

```rust
// Per decode step:
for layer in &self.layers {
    // INSIDE graph: local attention + FFN
    self.captured_layer_forward(graph, layer, hidden);

    // OUTSIDE graph: TP AllReduce
    self.comm.all_reduce_sum(&mut hidden, count, dtype, &self.stream);
}
```

This means we capture *per-layer* graphs (or sequences of layers), not one monolithic graph.

---

## 11. Incremental Detokenization

### The Problem

Streaming requires sending text to the client as tokens are generated. But:
- A single token may decode to an incomplete UTF-8 byte sequence (e.g., `[0xE4]` is the first byte of a 3-byte CJK character)
- BPE tokens can merge with the *next* token to form a different character
- Naïve `decode([new_token])` loses context and produces `\uFFFD` replacement characters

### The Solution: Surrogate Diff Detokenization

(Pattern from Mini-SGLang's `DetokenizeManager`)

```rust
pub struct IncrementalDetokenizer {
    tokenizer: Arc<Tokenizer>,
    /// Per-sequence state
    states: HashMap<u64, DetokenizeState>,
}

struct DetokenizeState {
    /// All token IDs generated so far
    token_ids: Vec<u32>,
    /// Full decoded string so far (may include partial chars not yet sent)
    full_text: String,
    /// Offset into token_ids: tokens before this were part of the last "safe" decode
    surr_offset: usize,
    /// Offset into token_ids: tokens before this have been read into full_text
    read_offset: usize,
    /// Number of characters already sent to the client
    sent_chars: usize,
}

impl IncrementalDetokenizer {
    /// Called after each token is generated. Returns the new text to send to the client,
    /// or empty string if the character isn't complete yet.
    pub fn add_token(&mut self, seq_id: u64, token_id: u32) -> String {
        let state = self.states.get_mut(&seq_id).unwrap();
        state.token_ids.push(token_id);

        // Decode two windows:
        // 1. "read" window: surr_offset → end (includes new token)
        // 2. "surr" window: surr_offset → read_offset (excludes new token)
        let read_ids = &state.token_ids[state.surr_offset..];
        let surr_ids = &state.token_ids[state.surr_offset..state.read_offset];

        let read_text = self.tokenizer.decode(read_ids, /*skip_special=*/ true);
        let surr_text = self.tokenizer.decode(surr_ids, /*skip_special=*/ true);

        // The new text is the diff: what read_text has beyond surr_text
        let new_text = if read_text.len() > surr_text.len() {
            &read_text[surr_text.len()..]
        } else {
            ""
        };

        // Only send if the text doesn't end with the replacement character (incomplete UTF-8)
        if !new_text.is_empty() && !new_text.ends_with('\u{FFFD}') {
            state.surr_offset = state.read_offset;
            state.read_offset = state.token_ids.len();
            state.full_text.push_str(new_text);
            let to_send = &state.full_text[state.sent_chars..];
            state.sent_chars = state.full_text.len();
            to_send.to_string()
        } else {
            // Character not complete — buffer and wait for next token
            state.read_offset = state.token_ids.len();
            String::new()
        }
    }

    pub fn finish(&mut self, seq_id: u64) -> String {
        // Flush any remaining buffered text
        let state = self.states.remove(&seq_id).unwrap();
        let remaining = self.tokenizer.decode(&state.token_ids[state.surr_offset..], true);
        remaining[state.full_text.len() - state.sent_chars..].to_string()
    }
}
```

### Batch Detokenization

For efficiency, batch the decode calls. The tokenizer crate supports `decode_batch`:

```rust
pub fn add_tokens_batch(&mut self, tokens: &[(u64, u32)]) -> Vec<(u64, String)> {
    // Collect read_ids and surr_ids for all sequences
    let (read_batch, surr_batch): (Vec<_>, Vec<_>) = tokens.iter()
        .map(|(seq_id, _)| {
            let s = &self.states[seq_id];
            (s.token_ids[s.surr_offset..].to_vec(),
             s.token_ids[s.surr_offset..s.read_offset].to_vec())
        })
        .unzip();

    let read_texts = self.tokenizer.decode_batch(&read_batch, true);
    let surr_texts = self.tokenizer.decode_batch(&surr_batch, true);

    // Compute diffs...
    // (same logic as single-token version, batched)
}
```

---

## 12. Pipeline Parallelism

*(Core design unchanged from v3 §8. Key refinement: the PP executor now uses the HAL `CommBackend` trait, and graph capture boundaries are per-layer to allow NCCL/HCCL between layers.)*

### PP Process Launch

```rust
/// Multi-process launcher: starts PP_SIZE processes, each with TP_SIZE threads.
pub fn launch(config: &Config) {
    // 1. Generate NCCL/HCCL unique ID on rank 0
    // 2. Broadcast ID to all ranks via TCP rendezvous
    // 3. Each process:
    //    a. Init device context for its TP GPUs
    //    b. Init TP communicator (intra-node NCCL/HCCL)
    //    c. Init PP communicator (inter-node or intra-node P2P)
    //    d. Load weights for its PP stage
    //    e. Warmup + allocate KV cache
    //    f. Capture graphs
    //    g. PP rank 0 starts HTTP server; others enter the executor loop
}
```

---

## 13. Expert Parallelism (MoE)

*(Core design unchanged from v3 §9. Refinement: added Ascend MoE comm type selection, expert stacking at load time.)*

### MoE Forward Pass (Refined)

```rust
async fn moe_forward(
    &self,
    hidden: &DeviceBuffer,      // [batch_tokens, hidden_dim]
    router_weight: &DeviceBuffer,
    experts: &DeviceBuffer,     // [num_experts_this_rank, out_dim, in_dim] (stacked)
    stream: &Stream,
) -> DeviceBuffer {
    // 1. Router: [batch_tokens, hidden_dim] × [hidden_dim, num_experts] → [batch_tokens, num_experts]
    let scores = self.kernels.gemm(&GemmParams {
        a: hidden, b: router_weight, ..
    }, stream);

    // 2. Top-K selection
    let (expert_ids, gate_weights) = self.kernels.topk(&scores, TOP_K_EXPERTS, stream);

    // 3. EP dispatch (if EP_SIZE > 1)
    let (local_hidden, local_expert_ids) = if EP_SIZE > 1 {
        self.comm_ep.all_to_all_dispatch(hidden, &expert_ids, stream).await
    } else {
        (hidden.clone(), expert_ids)
    };

    // 4. Grouped GEMM: compute all experts in a single batched call
    let expert_out = self.kernels.grouped_gemm(&GroupedGemmParams {
        inputs: &local_hidden,
        weights: experts,
        expert_ids: &local_expert_ids,
        ..
    }, stream);

    // 5. EP gather (if EP_SIZE > 1)
    let gathered = if EP_SIZE > 1 {
        self.comm_ep.all_to_all_gather(&expert_out, &local_expert_ids, stream).await
    } else {
        expert_out
    };

    // 6. Weighted sum
    self.kernels.weighted_scatter(&gathered, &gate_weights, stream)
}
```

---

## 14. Disaggregated Serving

*(Retained from v3 §10 with added detail on KV transfer sizing.)*

### KV Transfer Sizing

For a request with `S` tokens on a model with `L` layers (this PP rank's share), `H` KV heads per TP rank, and `D` head dim:

```
KV bytes = S × L × 2 × H × D × dtype_size

Example: S=32768, L=40 (PP=2), H=8 (TP=8 from 64 heads), D=128, BF16:
  = 32768 × 40 × 2 × 8 × 128 × 2
  = 10.7 GB per request
```

This means disaggregated mode is only practical with:
- **Intra-node IPC**: ~300 GB/s via NVLink → ~36ms for 10.7 GB ✓
- **Inter-node RDMA**: ~25 GB/s via InfiniBand → ~430ms for 10.7 GB — marginal
- **TCP**: ~3 GB/s → ~3.6s — too slow for long contexts

The generator should warn if `disagg: true` with inter-node PP and long `max_seq_len`.

---

## 15. Quantization Schemes

*(Unchanged from v3 §11. See v3 for CUDA and Ascend quant details.)*

---

## 16. Kernel Reuse Map

*(Unchanged from v3 §12.)*

---

## 17. Metrics and Observability

### 17.1 Per-Request Metrics

```rust
pub struct RequestMetrics {
    pub seq_id: u64,
    pub arrival_time: Instant,
    pub scheduled_time: Option<Instant>,    // first time it enters RUNNING
    pub first_token_time: Option<Instant>,  // first output token generated
    pub latest_token_time: Option<Instant>,
    pub finish_time: Option<Instant>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub finish_reason: Option<FinishReason>,
    pub prefix_cache_hit_tokens: usize,

    /// Inter-token latencies (for ITL percentile computation)
    pub token_timestamps: Vec<Instant>,
}

impl RequestMetrics {
    /// Time To First Token
    pub fn ttft(&self) -> Option<Duration> {
        Some(self.first_token_time? - self.arrival_time)
    }

    /// Time Per Output Token (average)
    pub fn tpot(&self) -> Option<Duration> {
        if self.generated_tokens <= 1 { return None; }
        let total = self.latest_token_time? - self.first_token_time?;
        Some(total / (self.generated_tokens as u32 - 1))
    }

    /// End-to-end latency
    pub fn e2e_latency(&self) -> Option<Duration> {
        Some(self.finish_time? - self.arrival_time)
    }

    /// Inter-Token Latencies (for percentile computation)
    pub fn itl_values(&self) -> Vec<Duration> {
        self.token_timestamps.windows(2)
            .map(|w| w[1] - w[0])
            .collect()
    }
}
```

### 17.2 System Metrics (Prometheus)

```rust
pub struct SystemMetrics {
    // Counters
    pub total_requests: Counter,
    pub total_prompt_tokens: Counter,
    pub total_generated_tokens: Counter,

    // Gauges
    pub running_requests: Gauge,
    pub waiting_requests: Gauge,
    pub swapped_requests: Gauge,
    pub gpu_kv_cache_usage: Gauge,        // 1.0 - (free_blocks / total_blocks)
    pub cpu_kv_cache_usage: Gauge,
    pub prefix_cache_hit_rate: Gauge,

    // Histograms
    pub ttft_seconds: Histogram,
    pub tpot_seconds: Histogram,
    pub itl_seconds: Histogram,
    pub e2e_latency_seconds: Histogram,
    pub prompt_tokens_histogram: Histogram,
    pub generated_tokens_histogram: Histogram,
}
```

### 17.3 Structured Logging

Use `tracing` with spans for request lifecycle:

```rust
#[instrument(skip_all, fields(seq_id = %seq_id, prompt_tokens = %prompt_len))]
async fn handle_request(seq_id: u64, prompt_len: usize) {
    tracing::info!("Request arrived");
    // ...
    tracing::info!(ttft_ms = ?metrics.ttft().map(|d| d.as_millis()), "First token");
    // ...
    tracing::info!(
        generated = metrics.generated_tokens,
        tpot_ms = ?metrics.tpot().map(|d| d.as_millis()),
        reason = ?metrics.finish_reason,
        "Request complete"
    );
}
```

### 17.4 Periodic Log Line

Every 10 seconds, log a summary:

```
INFO Throughput (in/out): 1234.5 / 4567.8 tok/s | Running: 45 | Waiting: 12 | KV cache: 78.3% | Prefix hit: 91.2%
```

---

## 18. Error Handling and Fault Tolerance

### 18.1 OOM During KV Allocation

```rust
fn handle_kv_oom(&mut self) -> Result<(), EngineError> {
    // Strategy: preempt running sequences to free blocks
    // 1. Sort running by priority (shortest generated count = lowest sunk cost)
    self.running.sort_by_key(|s| s.generated_tokens());

    // 2. Preempt until we have enough blocks
    while !self.kv_manager.has_free_blocks(MIN_RESERVE) {
        if let Some(victim) = self.running.pop() {
            tracing::warn!(seq_id = victim.id(), "Preempting sequence due to OOM");
            self.swap_out_to_cpu(&victim)?;
            self.swapped.push(victim);
        } else {
            // No one to preempt — fatal
            return Err(EngineError::OutOfMemory);
        }
    }
    Ok(())
}
```

### 18.2 NCCL/HCCL Failure

```rust
fn handle_comm_error(&self, err: CommError) -> EngineError {
    // NCCL errors are generally non-recoverable for the current batch.
    // Log, abort the batch, and try to reinitialize communication.
    tracing::error!(?err, "Communication backend error");

    match err {
        CommError::Timeout => {
            // Possibly a straggler GPU. Abort current batch, retry.
            EngineError::CommTimeout { retryable: true }
        }
        CommError::PeerFailed => {
            // A PP peer process died. Cannot recover without restart.
            EngineError::CommPeerDied
        }
        CommError::InternalError(msg) => {
            EngineError::CommInternal(msg)
        }
    }
}
```

### 18.3 Request Timeout

```rust
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

fn check_timeouts(&mut self) {
    let now = Instant::now();
    self.running.retain(|seq| {
        let elapsed = now - seq.arrival_time();
        if elapsed > seq.timeout().unwrap_or(DEFAULT_REQUEST_TIMEOUT) {
            tracing::warn!(seq_id = seq.id(), elapsed_s = elapsed.as_secs(), "Request timed out");
            seq.finish(FinishReason::Timeout);
            self.kv_manager.release(seq.block_ids());
            false
        } else {
            true
        }
    });
}
```

### 18.4 Malformed Request

Handle at the HTTP layer — never let bad input reach the engine:

```rust
async fn handle_chat_completion(
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // Validate
    if req.messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "messages array is empty".into() })));
    }
    if req.max_tokens.map_or(false, |m| m == 0) {
        return Err((StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "max_tokens must be > 0".into() })));
    }
    // Tokenize — catch tokenizer errors
    let tokens = tokenizer.encode(&prompt).map_err(|e| {
        (StatusCode::BAD_REQUEST,
         Json(ErrorResponse { error: format!("Tokenization failed: {e}") }))
    })?;
    if tokens.len() > MAX_SEQ_LEN {
        return Err((StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: format!("Prompt too long: {} > {}", tokens.len(), MAX_SEQ_LEN) })));
    }
    // ...
}
```

### 18.5 Graceful Shutdown

```rust
async fn shutdown(engine: Arc<Engine>) {
    tracing::info!("Shutting down — draining {} running requests", engine.running_count());

    // 1. Stop accepting new requests
    engine.stop_accepting();

    // 2. Wait for running requests to complete (with timeout)
    let drain_timeout = Duration::from_secs(60);
    tokio::select! {
        _ = engine.drain() => {
            tracing::info!("All requests drained");
        }
        _ = tokio::time::sleep(drain_timeout) => {
            tracing::warn!("Drain timeout — aborting remaining requests");
            engine.abort_all();
        }
    }

    // 3. Release resources
    engine.release_resources();
}
```

---

## 19. Key Rust Crates

| Crate | Purpose |
|---|---|
| `axum` | HTTP server, SSE streaming |
| `tokio` | Async runtime |
| `safetensors` | Load model weights (zero-copy memory mapping) |
| `tokenizers` | Tokenization + batch decode (HuggingFace) |
| `cudarc` | Safe Rust CUDA memory + stream + graph capture |
| `half` | BF16/FP16 types |
| `smallvec` | Inline storage for radix tree node children |
| `serde` / `serde_json` | Config parsing and API types |
| `tera` | Template engine for code generator |
| `clap` | CLI argument parsing |
| `tracing` / `tracing-subscriber` | Structured logging + spans |
| `metrics` + `metrics-exporter-prometheus` | Prometheus metrics |
| `bindgen` (build dep) | FFI bindings |
| `cmake` (build dep) | Kernel compilation |
| `parking_lot` | Fast mutex (for engine-level locks, not radix tree) |

**Ascend-specific:**

| Crate / Binding | Purpose |
|---|---|
| `acl-sys` (generated by bindgen) | Raw FFI to Ascend ACL runtime |
| `hccl-sys` (generated by bindgen) | Raw FFI to HCCL collective comm |

---

## 20. Implementation Phases with Test Plans

### Phase 1 — Generator + Single-GPU BF16 (3 weeks)

#### Deliverables

- [ ] `llm-gen` CLI: parses `config.yaml`, emits `config.rs` + `Cargo.toml` + `CMakeLists.txt`
- [ ] HAL traits defined; `CudaKernels` + `CudaBuffer` + `CudaStream`
- [ ] Kernel build system: cmake + bindgen for FlashAttention + BF16 cuBLAS
- [ ] `ModelConfig::from_hf` — reads `config.json` for Llama, Qwen, DeepSeek
- [ ] Safetensors weight loader with TP=1 (no sharding)
- [ ] Llama forward pass using HAL kernel dispatch (BF16)
- [ ] HTTP server (axum): `/v1/chat/completions`, non-streaming, single request

#### Test Plan

**T1.1 — Generator smoke test:**
```bash
cargo run --bin llm-gen -- configs/tp1-bf16-dev.yaml --output build/test-tp1
# Verify: build/test-tp1/src/config.rs contains TP_SIZE=1, PP_SIZE=1
# Verify: build/test-tp1/Cargo.toml has features=["cuda"]
# Verify: build/test-tp1/kernels/CMakeLists.txt includes gemm_bf16.cu
cd build/test-tp1 && cargo build --release  # must succeed
```

**T1.2 — Model config parsing:**
```rust
#[test]
fn test_llama3_config() {
    let cfg = ModelConfig::from_hf_file("test_fixtures/llama3-8b/config.json");
    assert_eq!(cfg.num_layers, 32);
    assert_eq!(cfg.num_q_heads, 32);
    assert_eq!(cfg.num_kv_heads, 8);  // GQA
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_experts, 0);   // dense model
}

#[test]
fn test_deepseek_v3_config() {
    let cfg = ModelConfig::from_hf_file("test_fixtures/deepseek-v3/config.json");
    assert!(cfg.num_experts > 0);  // MoE
    assert!(cfg.num_experts_per_tok > 0);
}
```

**T1.3 — Weight loading:**
```rust
#[test]
fn test_weight_count() {
    let weights = load_weights("test_fixtures/llama3-8b/", &cfg, 0, 1, 0, 1, &Device::Cpu);
    // Expected: embed + 32 layers × (attn_norm, qkv, o_proj, ffn_norm, gate, up, down) + final_norm + lm_head
    assert_eq!(weights.num_tensors(), 32 * 7 + 3);
}
```

**T1.4 — End-to-end single request:**
```bash
# Start server
./target/release/llm-server --model /models/llama3-8b --tp-rank 0

# In another terminal
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 32}'
# Verify: response contains a coherent answer mentioning "4"
# Verify: response.usage.prompt_tokens > 0
# Verify: response.usage.completion_tokens <= 32
```

**T1.5 — Correctness (logit comparison):**
```bash
# Generate logits for a known prompt and compare against reference (vLLM or HuggingFace)
python tools/compare_logits.py \
  --model llama3-8b \
  --prompt "The capital of France is" \
  --our-server http://localhost:8080 \
  --reference huggingface \
  --max-diff 0.01   # relative tolerance on top-1 logit
```

**Pass criteria:** All tests pass. Single-request latency < 5s for an 8B model on one GPU (not optimized yet).

---

### Phase 2 — KV Cache + Memory Profiling + Warmup (2 weeks)

#### Deliverables

- [ ] Memory profiling: measure model memory → warmup → compute available memory
- [ ] BlockPool with GPU pre-allocation
- [ ] RadixKvManager with find_prefix / insert_prefix / evict
- [ ] Warmup procedure before accepting requests
- [ ] PagedAttention kernel wired in

#### Test Plan

**T2.1 — Memory profiling accuracy:**
```rust
#[test]
fn test_memory_profiling() {
    // Load model, run warmup, allocate KV
    let (num_blocks, block_bytes) = run_memory_profiler(&config, &device);
    // After allocation, free memory should be ~(1 - utilization) × total
    let (free, total) = device.mem_info();
    let expected_headroom = total as f64 * (1.0 - config.gpu_memory_utilization);
    assert!((free as f64 - expected_headroom).abs() < 1e9); // within 1GB
}
```

**T2.2 — Block allocation/deallocation:**
```rust
#[test]
fn test_block_pool_lifecycle() {
    let pool = BlockPool::new(&device, 100, ...);
    assert_eq!(pool.free_count(), 100);

    let blocks = pool.allocate(10);
    assert_eq!(blocks.len(), 10);
    assert_eq!(pool.free_count(), 90);

    pool.free(&blocks);
    assert_eq!(pool.free_count(), 100);
}
```

**T2.3 — Prefix cache hit rate:**
```bash
# Send 20 requests with the same system prompt (1024 tokens) + different user messages
python tools/prefix_cache_test.py \
  --server http://localhost:8080 \
  --system-prompt "You are a helpful assistant..." \
  --num-requests 20

# Verify: prefix_cache_hit_rate metric > 0.90
curl http://localhost:8080/metrics | grep prefix_cache_hit_rate
# Expected: prefix_cache_hit_rate >= 0.90
```

**T2.4 — Radix tree correctness:**
```rust
#[test]
fn test_radix_prefix_match() {
    let mut kv = RadixKvManager::new_cpu(100);
    let tokens_a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // 1 block
    let blocks_a = kv.allocate_blocks(1);
    kv.insert_prefix(&tokens_a, &blocks_a);

    // Exact match
    let (cached, ids) = kv.find_prefix(&tokens_a);
    assert_eq!(cached, 16);
    assert_eq!(ids, blocks_a);

    // Prefix match
    let tokens_b = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
    let (cached, ids) = kv.find_prefix(&tokens_b);
    assert_eq!(cached, 16);  // first block matches
    assert_eq!(ids.len(), 1);

    // No match
    let tokens_c = vec![99, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let (cached, _) = kv.find_prefix(&tokens_c);
    assert_eq!(cached, 0);
}

#[test]
fn test_radix_eviction() {
    let mut kv = RadixKvManager::new_cpu(2); // only 2 blocks
    // Fill both blocks
    let tokens1 = vec![1; 16];
    let tokens2 = vec![2; 16];
    let b1 = kv.allocate_blocks(1);
    let b2 = kv.allocate_blocks(1);
    kv.insert_prefix(&tokens1, &b1);
    kv.insert_prefix(&tokens2, &b2);
    kv.release(&b1); // ref_count → 0 for block 1
    kv.release(&b2); // ref_count → 0 for block 2

    // Pool is full. Evict should free the oldest.
    let freed = kv.evict(1);
    assert_eq!(freed, 1);
    assert_eq!(kv.gpu_pool.free_count(), 1);
}
```

**Pass criteria:** Memory utilization within 5% of target. Prefix cache hit > 90% for repeated system prompts.

---

### Phase 3 — Continuous Batching + Streaming + Overlap Scheduler (3 weeks)

#### Deliverables

- [ ] Overlap scheduler with token-budget scheduling
- [ ] Chunked prefill
- [ ] SSE token streaming
- [ ] Incremental detokenization
- [ ] Preemption + CPU swap
- [ ] Request timeout handling
- [ ] Prometheus metrics endpoint

#### Test Plan

**T3.1 — Streaming correctness:**
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Count from 1 to 10"}], "max_tokens": 64, "stream": true}'
# Verify: SSE events arrive incrementally (not all at once)
# Verify: reassembled text matches non-streaming result
# Verify: no \uFFFD replacement characters in output
```

**T3.2 — Incremental detokenization (CJK / multi-byte):**
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Translate to Chinese: Hello, how are you?"}], "max_tokens": 64, "stream": true}'
# Verify: Chinese characters stream correctly without mojibake
# Verify: no partial UTF-8 sequences in any SSE event
```

**T3.3 — Concurrent load test:**
```bash
python tools/load_test.py \
  --server http://localhost:8080 \
  --concurrency 50 \
  --num-requests 200 \
  --prompt-len 512 \
  --max-tokens 128

# Pass criteria:
#   - All 200 requests complete without error
#   - p50 TTFT < 200ms
#   - p99 TTFT < 2s
#   - No OOM crashes
#   - Throughput > 500 tok/s (single GPU, 8B model)
```

**T3.4 — Preemption works:**
```bash
# Send many long-output requests to exhaust KV cache
python tools/oom_test.py \
  --server http://localhost:8080 \
  --concurrency 100 \
  --max-tokens 4096 \
  --prompt-len 2048

# Verify: no crashes (preemption and swap should kick in)
# Verify: all requests eventually complete
# Verify: logs show "Preempting sequence" messages
```

**T3.5 — Metrics endpoint:**
```bash
curl http://localhost:8080/metrics | grep -E "^(ttft|tpot|running|waiting|kv_cache)"
# Verify: all expected metrics are present and have reasonable values
```

**T3.6 — Overlap scheduling benefit:**
```bash
# Run same load test with overlap=true and overlap=false
# Compare p50 TTFT and throughput
# Verify: overlap=true has at least 10% better throughput
```

**Pass criteria:** Stable under 50 concurrent requests. Streaming works with multi-byte characters. Preemption prevents OOM. Metrics are accurate.

---

### Phase 4 — TP=8 (1 week)

#### Deliverables

- [ ] NCCL intra-node init
- [ ] TP weight sharding at load time (dim-0, dim-1, KV head special case)
- [ ] Per-GPU worker threads
- [ ] AllReduce after attention and FFN
- [ ] CUDA graph capture for decode

#### Test Plan

**T4.1 — Weight sharding correctness:**
```rust
#[test]
fn test_tp_weight_shapes() {
    let cfg = ModelConfig { num_q_heads: 32, num_kv_heads: 8, hidden_size: 4096, head_dim: 128, .. };
    let weights_r0 = load_weights(&model_dir, &cfg, /*tp_rank=*/ 0, /*tp_size=*/ 8, ..);
    let weights_r1 = load_weights(&model_dir, &cfg, /*tp_rank=*/ 1, /*tp_size=*/ 8, ..);

    // Q proj: 4096 → 4096, sharded dim-0: each rank gets 4096/8 = 512
    assert_eq!(weights_r0.layer(0).q_proj.shape(), [512, 4096]);
    // K proj: 8 KV heads / 8 TP = 1 head per rank → 128
    assert_eq!(weights_r0.layer(0).k_proj.shape(), [128, 4096]);
    // O proj: sharded dim-1: each rank gets 4096/8 = 512 columns
    assert_eq!(weights_r0.layer(0).o_proj.shape(), [4096, 512]);
}
```

**T4.2 — Logit equivalence (TP=1 vs TP=8):**
```bash
# Same prompt, compare top-5 logits between TP=1 and TP=8 builds
python tools/compare_logits.py \
  --model llama3-70b \
  --prompt "The meaning of life is" \
  --tp1-server http://host-tp1:8080 \
  --tp8-server http://host-tp8:8080 \
  --max-diff 0.02
```

**T4.3 — Throughput on 70B model:**
```bash
python tools/benchmark.py \
  --server http://localhost:8080 \
  --model llama3-70b \
  --prompt-len 512 \
  --max-tokens 128 \
  --concurrency 32

# Pass criteria:
#   - Decode throughput > 3,000 tok/s (BF16) on 8×H100
#   - No NCCL errors in logs
```

**T4.4 — CUDA graph decode speedup:**
```bash
# Compare throughput: enforce_eager=true vs enforce_eager=false
# Graph mode should be at least 1.3× faster for small batch decode
python tools/benchmark.py --enforce-eager true ...  → baseline
python tools/benchmark.py --enforce-eager false ... → graph
# Verify: graph throughput / baseline throughput > 1.3
```

**Pass criteria:** Llama-3.1-70B on 8×H100 at >3,000 tok/s (BF16). TP=1 and TP=8 produce equivalent logits.

---

### Phase 5 — Ascend NPU Backend (3 weeks)

#### Deliverables

- [ ] `acl-sys` and `hccl-sys` bindgen from CANN headers
- [ ] `AscendKernels` implementing `KernelDispatch`
- [ ] `AscendBuffer` variant in `DeviceBuffer` enum
- [ ] `AscendStream` variant in `Stream` enum
- [ ] `HcclComm` implementing `CommBackend`
- [ ] KV cache block shape adjusted for Ascend 910 layout
- [ ] ACL graph capture via `dlinfer` pattern
- [ ] Bucketing for Ascend-optimal batch sizes
- [ ] Generator: `hardware: ascend` emits CANN cmake + correct config

#### Test Plan

**T5.1 — Single-GPU forward pass on Ascend:**
```bash
./target/release/llm-server --model /models/qwen-7b --tp-rank 0 --hardware ascend
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 32}'
# Verify: coherent response
```

**T5.2 — TP=8 on Ascend 910B:**
```bash
# Launch 8 processes with HCCL
python tools/benchmark.py \
  --server http://localhost:8080 \
  --model qwen-72b \
  --prompt-len 512 \
  --max-tokens 128 \
  --concurrency 16

# Pass criteria:
#   - Decode throughput > 2,000 tok/s (BF16) on 8×910B
#   - No HCCL errors
```

**T5.3 — Cross-hardware logit comparison:**
```bash
python tools/compare_logits.py \
  --model qwen-7b \
  --prompt "The speed of light is" \
  --cuda-server http://cuda-host:8080 \
  --ascend-server http://ascend-host:8080 \
  --max-diff 0.05  # wider tolerance due to different numerics
```

**T5.4 — ACL graph capture:**
```bash
# Verify graph capture succeeds and improves decode throughput
# Benchmark with and without enforce_eager on Ascend
```

**Pass criteria:** Qwen-72B serving on 8×Ascend 910B with correct output. HCCL TP works. ACL graph capture provides speedup.

---

### Phase 6 — FP8 Quantization (1 week)

#### Deliverables

- [ ] FP8 kernel variant for CUDA (CUTLASS) and Ascend (CANN `aclnnMatMulFp8`)
- [ ] Generator emits FP8 config + kernel selection
- [ ] Weight loader: BF16 → FP8 conversion or pre-quantized loading
- [ ] Per-channel / per-tensor scale tensors

#### Test Plan

**T6.1 — FP8 quality check:**
```bash
# Compare perplexity on a validation set: BF16 vs FP8
python tools/eval_perplexity.py \
  --model llama3-70b \
  --dataset wikitext-2 \
  --bf16-server http://bf16-host:8080 \
  --fp8-server http://fp8-host:8080

# Pass criteria: FP8 perplexity within 0.5 of BF16 perplexity
```

**T6.2 — FP8 throughput:**
```bash
python tools/benchmark.py \
  --server http://fp8-host:8080 \
  --model llama3-70b \
  --prompt-len 512 --max-tokens 128 --concurrency 32

# Pass criteria: > 4,500 tok/s (>1.3× over BF16's 3,000)
```

**T6.3 — FP8 on Ascend:**
```bash
# Same tests as T6.1 and T6.2 on Ascend 910B
# Pass criteria: FP8 working, >1.3× throughput over BF16
```

---

### Phase 7 — Pipeline Parallelism (2 weeks)

#### Deliverables

- [ ] `PipelineExecutor` with micro-batch formation
- [ ] `CommBackend::send/recv` for inter-stage activation transfer
- [ ] SSH/MPI process launcher
- [ ] TCP rendezvous for NCCL/HCCL unique ID broadcast
- [ ] Generator: `pp > 1` emits pipeline executor; `pp == 1` emits simple executor

#### Test Plan

**T7.1 — PP correctness:**
```bash
# Compare logits: TP=8 PP=1 vs TP=8 PP=2 (both BF16, same model)
# They should produce identical logits (PP only changes which rank computes which layers)
python tools/compare_logits.py \
  --model llama3-70b \
  --prompt "Hello world" \
  --pp1-server http://pp1-host:8080 \
  --pp2-server http://pp2-host:8080 \
  --max-diff 0.001  # should be nearly identical
```

**T7.2 — DeepSeek-V3 on 2×8×H100:**
```bash
python tools/benchmark.py \
  --server http://pp-host:8080 \
  --model deepseek-v3 \
  --prompt-len 1024 --max-tokens 256 --concurrency 16

# Pass criteria:
#   - Decode throughput > 1,500 tok/s
#   - PP bubble overhead < 5% (measure: time_in_nccl_wait / total_step_time)
```

**T7.3 — PP with prefix cache:**
```bash
# Verify prefix cache works across PP ranks (same tokens → same cache hit on all ranks)
python tools/prefix_cache_test.py \
  --server http://pp-host:8080 \
  --num-requests 20
# Verify: prefix_cache_hit_rate > 0.90
```

---

### Phase 8 — Expert Parallelism for MoE (2 weeks)

#### Deliverables

- [ ] `CommBackend::all_to_all` for NCCL and HCCL
- [ ] `MoeLayer` with grouped GEMM + AllToAll dispatch/gather
- [ ] Expert stacking at weight load time
- [ ] Generator: `ep > 1` emits EP-aware MoE executor
- [ ] Ascend MoE comm type selection (MC2 vs AllToAll vs AllGather)

#### Test Plan

**T8.1 — MoE routing correctness:**
```bash
# Compare DeepSeek-V3 output: our server vs reference (vLLM or lmdeploy)
python tools/compare_logits.py \
  --model deepseek-v3 \
  --prompt "Explain quantum computing" \
  --our-server http://moe-host:8080 \
  --reference vllm \
  --max-diff 0.05
```

**T8.2 — EP scaling:**
```bash
# Benchmark EP=1 vs EP=8 (same total GPUs, just different parallelism)
# EP=8 should improve throughput for memory-bound MoE models
python tools/benchmark.py --ep 1 ...  → baseline
python tools/benchmark.py --ep 8 ...  → ep
# Verify: EP routing overhead < 8% (measure all_to_all time / total step time)
```

**T8.3 — Ascend MoE comm type:**
```bash
# Verify the correct comm type is selected per SOC version
# Test with different max_tokens thresholds
# Verify logs show "MoE comm type: MC2" or "AllToAll" as expected
```

---

### Phase 9 — W8A8, W4A16, Disaggregated Serving (2 weeks)

#### Deliverables

- [ ] W8A8 kernel variant (CUTLASS + CANN)
- [ ] W4A16 AWQ variant
- [ ] Disaggregated prefill/decode: `--role prefill|decode`, KV IPC transfer
- [ ] HTTP router for disagg mode

#### Test Plan

**T9.1 — W4A16 memory savings:**
```bash
# Fit a 405B model on 8 GPUs with W4A16 (wouldn't fit in BF16)
./target/release/llm-server --model llama3-405b --quant w4a16
# Verify: model loads without OOM
# Verify: correct output for simple prompts
```

**T9.2 — Disaggregated mode latency:**
```bash
# Compare TTFT: integrated mode vs disagg mode
# Disagg should have better TTFT under mixed workload (long prefills + concurrent decodes)
python tools/disagg_benchmark.py \
  --prefill-server http://prefill-host:8080 \
  --decode-server http://decode-host:8080 \
  --mixed-workload  # some long prompts (8K+) mixed with short prompts

# Pass criteria:
#   - Disagg p50 TTFT < integrated p50 TTFT (under mixed load)
#   - All requests produce correct output
#   - KV transfer time logged and < 100ms for intra-node
```

**T9.3 — W8A8 on Ascend:**
```bash
python tools/benchmark.py \
  --server http://ascend-w8a8:8080 \
  --model qwen-72b \
  --quant w8a8
# Pass criteria: >1.3× throughput over BF16
```

---

## 21. Directory Layout

```
llm-inference/
├── llm-gen/                          ← code generator
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── emit_config.rs
│   │   ├── emit_cargo.rs
│   │   ├── emit_cmake.rs             ← CUDA cmake
│   │   ├── emit_cmake_cann.rs        ← Ascend cmake
│   │   └── emit_main.rs
│   └── templates/
│       ├── config.rs.tera
│       ├── Cargo.toml.tera
│       ├── CMakeLists.txt.tera
│       └── CMakeLists_cann.txt.tera
│
├── llm-server/                       ← server library
│   ├── Cargo.toml
│   ├── kernels/
│   │   ├── cuda/                     ← CUDA kernel wrappers
│   │   │   ├── attention.cu
│   │   │   ├── layernorm.cu
│   │   │   ├── pos_encoding.cu
│   │   │   ├── sampling.cu
│   │   │   ├── gemm_bf16.cu
│   │   │   ├── gemm_fp8.cu
│   │   │   ├── gemm_w8a8.cu
│   │   │   └── gemm_w4a16.cu
│   │   ├── ascend/                   ← Ascend wrappers
│   │   │   ├── attention.cpp
│   │   │   ├── layernorm.cpp
│   │   │   ├── gemm_bf16.cpp
│   │   │   ├── gemm_w8a8.cpp
│   │   │   └── gemm_fp8.cpp
│   │   └── include/
│   │       ├── llm_kernels.h
│   │       └── llm_kernels_ascend.h
│   └── src/
│       ├── hal/                      ← Hardware Abstraction Layer
│       │   ├── mod.rs                ← KernelDispatch, CommBackend traits + DeviceBuffer enum
│       │   ├── cuda.rs
│       │   └── ascend.rs
│       ├── api/
│       │   ├── routes.rs             ← HTTP handlers with validation
│       │   └── types.rs              ← OpenAI-compatible API types
│       ├── config/
│       │   ├── hf_config.rs          ← HfConfig + ModelConfig::from_hf
│       │   └── runtime.rs            ← RuntimeConfig
│       ├── weights/
│       │   ├── loader.rs             ← safetensors discovery + streaming load
│       │   ├── shard.rs              ← TP sharding rules
│       │   └── fuse.rs               ← QKV fusion, expert stacking
│       ├── memory/
│       │   ├── profiler.rs           ← warmup + memory profiling
│       │   └── allocator.rs          ← KV cache block count computation
│       ├── scheduler/
│       │   ├── mod.rs
│       │   ├── overlap.rs            ← OverlapScheduler
│       │   ├── batch.rs              ← token budget + layout tracking
│       │   ├── preempt.rs            ← preemption + swap
│       │   └── queue.rs
│       ├── kv_cache/
│       │   ├── radix_tree.rs         ← RadixKvManager
│       │   ├── block_pool.rs
│       │   └── swap.rs
│       ├── graph/
│       │   ├── capture.rs            ← graph batch sizes + pool reuse
│       │   └── replay.rs             ← padding + graph dispatch
│       ├── detokenize/
│       │   └── incremental.rs        ← IncrementalDetokenizer
│       ├── executor/
│       │   ├── simple.rs             ← TP-only
│       │   ├── pipeline.rs           ← PP executor
│       │   ├── moe.rs                ← MoE + EP layer
│       │   ├── worker.rs
│       │   └── comm.rs
│       ├── models/
│       │   ├── mod.rs
│       │   ├── llama.rs
│       │   ├── deepseek.rs           ← MoE model
│       │   ├── qwen.rs
│       │   └── weights.rs
│       ├── disagg/                   ← Disaggregated serving
│       │   ├── prefill_engine.rs
│       │   ├── decode_engine.rs
│       │   └── kv_transfer.rs
│       ├── metrics/
│       │   ├── request.rs            ← RequestMetrics
│       │   ├── system.rs             ← SystemMetrics (Prometheus)
│       │   └── logging.rs            ← Periodic log line
│       └── error.rs                  ← EngineError enum + handlers
│
├── tools/                            ← Test + benchmark scripts
│   ├── compare_logits.py
│   ├── load_test.py
│   ├── prefix_cache_test.py
│   ├── oom_test.py
│   ├── eval_perplexity.py
│   ├── disagg_benchmark.py
│   └── benchmark.py
│
├── test_fixtures/                    ← Model config.json samples for unit tests
│   ├── llama3-8b/config.json
│   ├── deepseek-v3/config.json
│   └── qwen-72b/config.json
│
├── configs/
│   ├── tp8-fp8-cuda-single-node.yaml
│   ├── tp8-pp2-fp8-cuda-two-node.yaml
│   ├── tp8-w8a8-ascend-single-node.yaml
│   ├── tp4-w4a16-4gpu.yaml
│   ├── tp8-ep8-fp8-moe-single-node.yaml
│   └── tp1-bf16-dev.yaml
│
└── build/                            ← generated output (gitignored)
```

---

## 22. Performance Targets

### Llama-3.1-70B on 8×H100 (FP8, TP=8)

| Metric | Target | How Measured |
|---|---|---|
| Decode throughput | > 4,500 tok/s | `benchmark.py --concurrency 32` |
| TTFT (512-tok prompt, cold) | < 80ms | p50 from load test |
| TTFT (512-tok prompt, cached) | < 15ms | p50 after cache warm |
| Prefix cache hit rate | > 85% | Prometheus metric |
| Overlap scheduling gain | > 15% throughput | A/B test overlap=true vs false |

### DeepSeek-V3 (671B MoE) on 2×8×H100 (FP8, TP=8, PP=2, EP=8)

| Metric | Target | How Measured |
|---|---|---|
| Decode throughput | > 1,500 tok/s | `benchmark.py --concurrency 16` |
| TTFT | < 300ms | p50 from load test |
| PP bubble overhead | < 5% | `nccl_wait_time / total_step_time` from tracing |
| EP routing overhead | < 8% | `all_to_all_time / total_step_time` from tracing |

### Qwen-72B on 8×Ascend 910B (W8A8, TP=8)

| Metric | Target | How Measured |
|---|---|---|
| Decode throughput | > 3,000 tok/s | `benchmark.py --concurrency 16` |
| TTFT (512-tok prompt, cold) | < 120ms | p50 from load test |
| TTFT (512-tok prompt, cached) | < 25ms | p50 after cache warm |
| Memory utilization | > 80% of device HBM | Prometheus metric |

---

## Key Design Decisions — Rationale

**Concrete buffer enum vs trait object (v4 change):** v3's `dyn DeviceBuffer` added a vtable indirection on every `as_ptr()` call — tens of millions of times per forward pass for attention metadata construction. An enum with two variants (`Cuda` / `Ascend`) is const-propagated by the compiler when `HARDWARE` is a compile-time constant, giving zero-cost abstraction. Trait objects are only used for `KernelDispatch` and `CommBackend`, which are called once per kernel launch (microsecond+ granularity).

**Radix tree without per-node RwLock (v4 change):** v3's `Arc<RwLock<RadixNode>>` per node creates N lock acquisitions per prefix lookup (N = prefix length in blocks). Since the scheduler is single-threaded and owns the radix tree, we use direct `&mut` access. The tree is only shared across the scheduler → no concurrent readers/writers.

**Warmup BEFORE KV allocation (v4 addition):** Warmup runs a forward pass that allocates transient buffers (activations, scratch space). Peak memory during warmup reveals how much headroom the model needs. Without warmup first, the KV cache allocation would overcommit and the first real request would OOM.

**Graph pool reuse (v4 addition):** Without pool reuse, each `CUDAGraph::capture()` allocates a new internal workspace. For 20+ batch sizes, this wastes hundreds of MB. By capturing the largest batch first and reusing `.pool()`, all subsequent graphs share the same workspace.

**Incremental detokenization via surrogate diff (v4 addition):** The key insight (from Mini-SGLang) is to decode *two overlapping windows* of tokens and take the difference. This correctly handles multi-byte characters and BPE merge boundaries without any special-casing per tokenizer type.

---

*v4 is a buildable specification. Each phase has concrete deliverables, and each deliverable has a test that can be run to verify it. The test plans are designed to catch the bugs that actually occur in LLM serving: wrong TP sharding, memory overcommit, graph capture failures, and streaming encoding errors.*
