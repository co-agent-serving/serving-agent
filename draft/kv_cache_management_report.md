# KV Cache Management for LLM Inference: A Comprehensive Report

> **Context:** This report synthesizes a deep-dive discussion on vLLM internals, Ray integration, RLHF frameworks, and modern KV cache architectures for production-grade agentic inference systems.

---

## Table of Contents

1. [Background: vLLM & Ray Integration](#1-background-vllm--ray-integration)
2. [Why Ray? The Real Role in vLLM](#2-why-ray-the-real-role-in-vllm)
3. [Weight Synchronization in RLHF Frameworks](#3-weight-synchronization-in-rlhf-frameworks)
4. [KV Cache Fundamentals](#4-kv-cache-fundamentals)
5. [Automatic Prefix Caching (APC)](#5-automatic-prefix-caching-apc)
6. [Hierarchical KV Storage Tiers](#6-hierarchical-kv-storage-tiers)
7. [LMCache: The KV Connector Layer](#7-lmcache-the-kv-connector-layer)
8. [Prefill-Decode (PD) Disaggregation](#8-prefill-decode-pd-disaggregation)
9. [Mooncake: KVCache-Centric Architecture](#9-mooncake-kvcache-centric-architecture)
10. [KV-Cache Aware Routing at Scale](#10-kv-cache-aware-routing-at-scale)
11. [Full Production Stack](#11-full-production-stack)
12. [Practical Recommendations](#12-practical-recommendations)

---

## 1. Background: vLLM & Ray Integration

### Executor Backend Selection

vLLM supports two distributed execution backends:

| Backend | Flag | Best For |
|---|---|---|
| **Multiprocessing** | `mp` | Single-node, multi-GPU (default) |
| **Ray** | `ray` | Multi-node, or inside a Ray cluster |

The executor is chosen automatically:

- **Multiprocessing** is the default when running on a single node with sufficient GPUs and not inside a Ray placement group.
- **Ray** is used automatically when: (a) vLLM detects it is already inside a Ray placement group (e.g., deployed via Ray Serve), or (b) multi-node is required.

Override explicitly:

```python
llm = LLM("meta-llama/Llama-3.1-70B",
           tensor_parallel_size=4,
           distributed_executor_backend="ray")  # or "mp"
```

### Key Source Files

| File | Role |
|---|---|
| `vllm/executor/ray_distributed_executor.py` | Main `RayDistributedExecutor` class |
| `vllm/executor/ray_utils.py` | `RayWorkerWrapper` Ray actor, placement group utilities |
| `vllm/v1/executor/ray_utils.py` | V1 engine equivalent |

### Ray Placement Groups

When Ray is used, vLLM creates a **placement group** — one bundle per GPU worker — to atomically reserve resources before spawning any workers. Each `RayWorkerWrapper` actor is pinned to a specific bundle via `PlacementGroupSchedulingStrategy`. This allows Ray to set `CUDA_VISIBLE_DEVICES` correctly before the worker loads model weights.

Placement strategies:
- **`STRICT_PACK`** — all actors on the same node (single-node TP)
- **`SPREAD`** — bundles distributed across nodes (multi-node PP)

### When Ray is Truly Required

| Scenario | Ray needed? |
|---|---|
| Single-node TP (≤8 GPUs) | No — `mp` is better |
| Multi-node TP or PP | **Yes — only option** |
| Pipeline parallelism > 1 node | **Yes — mandatory** |
| Running inside Ray Serve | **Yes — auto-detected** |
| RLHF GPU co-location (veRL, OpenRLHF) | **Yes — essential** |
| CPU backend | No — always `mp` |

### CPU Backend: Why Ray Is Excluded

The CPU backend does not support Ray for deliberate design reasons:
1. **Ray's executor is GPU-centric** — CPU affinity and NUMA topology control require fundamentally different resource management.
2. **Ray's background processes contend with CPU inference threads** — Ray's logger and scheduling processes consume the same CPU cores needed for inference.
3. **Native `mp` already covers intra-node cases** — TP, PP, and DP all work without Ray on a single node.
4. **Inter-node CPU cases aren't compelling** — Gloo-based CPU collective communication is too slow for TP/PP across nodes; inter-node DP is better handled via Kubernetes.

---

## 2. Why Ray? The Real Role in vLLM

Ray's value is not in how tensors move between GPUs (that is NCCL's job), but in **orchestrating the cluster** before inference starts.

### Unique Contributions

**Multi-node worker bootstrap:** `fork`/`spawn` is fundamentally a single-host mechanism. Ray is the only mechanism in vLLM that can create and manage Python worker processes on remote machines. Without Ray, vLLM cannot cross node boundaries.

**Atomic resource reservation:** Ray placement groups reserve GPUs across N machines before any model weights are loaded, preventing partial failures mid-initialization.

**Ray Serve integration:** When deployed via Ray Serve, vLLM is already inside a Ray actor. The `RayDistributedExecutor` detects this via `ray.util.get_current_placement_group()` and participates in Ray's deployment and autoscaling infrastructure natively.

**RLHF co-location (the killer use case):** In RLHF frameworks, training actors (FSDP/Megatron) and inference workers (vLLM) must share the same GPU pool and synchronize weights after every training step. Ray's placement group API is the mechanism that makes GPU resource sharing between these otherwise independent processes possible.

---

## 3. Weight Synchronization in RLHF Frameworks

### The Core Problem

After a training step, model weights in the training process have changed via gradient updates, but the vLLM inference engine still holds the old weights. They are separate Python processes — possibly on separate machines — with no direct memory access to each other.

**Ray's role:** Ray does not move the weights itself. It is the coordination layer that sets up the communication channel and orchestrates the synchronization call. The actual data movement is done by NCCL or CUDA IPC.

### Three Synchronization Mechanisms

#### 1. NCCL Broadcast (multi-node, disaggregated)

The most common pattern for separate training and inference GPUs.

```python
# On the training side (rank 0):
handle = llm.collective_rpc.remote("update_weight", args=(name, dtype, shape))
model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
ray.get(handle)  # barrier: wait for all vLLM workers to finish
```

- `llm.collective_rpc.remote(...)` is a **Ray remote call** that fans out simultaneously to all `RayWorkerWrapper` actors.
- `ray.get(handle)` is the barrier that prevents the training loop from proceeding until all workers have loaded new weights.
- A `stateless_init_process_group` creates a NCCL communicator spanning both the trainer and all vLLM workers.

#### 2. CUDA IPC (same-node, co-located)

When training and inference are colocated on the same GPU, the trainer exposes a GPU tensor's memory address via a CUDA IPC handle — a zero-copy transfer. The vLLM worker maps that memory directly into its address space without copying data.

Ray placement groups with `STRICT_PACK` strategy enforce that trainer and vLLM worker are on the same physical node.

#### 3. In-Memory Resharding (veRL's HybridEngine)

veRL avoids the network entirely. Actor and rollout live in the same process or placement group. The tricky part: training uses one parallelism layout (e.g., FSDP sharded across 8 GPUs) while inference uses another (e.g., TP=4). veRL's HybridEngine performs a 3D tensor redistribution — resharding in-place — before handing weights to vLLM.

### What Ray Does in Each Case

| Function | Ray's Mechanism |
|---|---|
| Fan-out to all workers simultaneously | `collective_rpc.remote(...)` |
| Synchronization barrier | `ray.get(handle)` |
| Enforce co-location for IPC | Placement group with `STRICT_PACK` |
| Cross-process group setup | `stateless_init_process_group` via Ray actor RPC |

### RLHF Framework Comparison

| | **veRL** | **OpenRLHF** | **vLLM baseline** |
|---|---|---|---|
| Architecture | Co-located monolith (`WorkerDict`) | Separate resource pools | Separated GPUs |
| Training backend | FSDP / Megatron | DeepSpeed | Raw PyTorch |
| Weight sync | In-memory (shared process) | NCCL / CUDA IPC | Ray collective RPC |
| Ray placement group | Unified group for all models | Per-module or shared groups | Explicit user-created pg |
| vLLM role | Rollout engine, co-located | Dedicated inference workers | Dedicated inference actors |

### New Native Weight Sync API (vLLM v1)

Previously each framework (SkyRL, veRL, TRL) maintained its own weight syncing infrastructure. A standardized API is now being integrated:

```python
# New standardized flow (vLLM v1)
inference_handle = llm.init_weight_transfer_engine.remote(
    dict(init_info=dict(master_address=..., master_port=..., rank_offset=1, world_size=...))
)
train_handle = train_model.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])  # both sides ready simultaneously
```

---

## 4. KV Cache Fundamentals

### Why KV Cache Matters for Agents

Agentic workloads represent the most extreme case of prefix dominance. An agent's context contains the agent's goals, tool definitions, and a long history of actions and observations. This prefix grows with every turn. Reusing it via caching is essential for agents to be computationally viable at scale.

> The KV-cache hit rate is the single most important metric for a production-stage AI agent — it directly impacts both latency and cost.

### PagedAttention: The Foundation

vLLM's PagedAttention partitions the KV cache of each request into **KV Blocks**. Each block contains the attention keys and values for a fixed number of tokens (16 by default). Blocks are stored in non-contiguous physical memory, eliminating fragmentation.

Key design choices:
- **Block size of 16 tokens** is deliberate — larger blocks reduce prefix hit rates (unless complex partial-matching is added) and increase fragmentation.
- **Non-contiguous allocation** means memory is allocated on demand, preventing the large pre-allocated buffers that wasted GPU RAM in earlier systems.

---

## 5. Automatic Prefix Caching (APC)

### Core Mechanism

Each KV block is uniquely identified by a hash of:
- The tokens within the block
- All tokens in the prefix before the block

When a new request arrives, vLLM walks its prefix block-by-block. Any block whose hash matches a cached block is reused directly — skipping its recomputation.

### Extensions

**Multi-LoRA serving:** The hash includes the LoRA adapter ID, enabling transparent prefix caching across all adapters simultaneously. This simplifies system implementation and improves global cache hit rates.

**Multi-modal models:** Different hashing strategies handle non-text input modalities (images, audio) within the same block addressing scheme.

### Critical Pitfall: Context Truncation

Context truncation — widely used in industry to cap prompt length — can **reduce prefix cache hit ratio by half**. If the system prompt or tool history is truncated differently on each request, hash matches break. Avoid truncation strategies that modify the stable prefix head of the context.

---

## 6. Hierarchical KV Storage Tiers

vLLM's native KV cache lookup follows a tiered fallback:

```
1. GPU HBM (fastest, smallest)
   ↓ miss
2. CPU DRAM (large, ~10× more capacity)
   ↓ miss
3. KV Connectors (external storage: disk, Redis, remote nodes)
```

This hierarchy allows inference engines to serve from warm caches in GPU memory while gracefully degrading to slower tiers on cold starts.

---

## 7. LMCache: The KV Connector Layer

LMCache is an external caching system that extends vLLM's native tiering into a full enterprise-grade KV management layer. It sits between the inference engine and storage backends.

### Capabilities

- **Context caching:** KV cache offloading and sharing across queries and sessions
- **PD disaggregation:** Cross-engine KV transfer for prefill/decode separation
- **Multi-backend storage:** CPU DRAM, local NVMe, remote NVMe, Redis

### Performance

Combining LMCache with vLLM achieves up to **15× improvement in throughput** on workloads such as multi-round question answering and document analysis.

### Architecture

LMCache introduces a standardized KV connector interface that decouples KV cache management from the inference engine. This design — developed collaboratively between the LMCache and vLLM teams — ensures compatibility regardless of how vLLM's internal architecture evolves.

```
vLLM Engine
  ↓  KV Connector API
LMCache (caching logic, eviction, PD proxy)
  ↓  backend driver
CPU DRAM / Local SSD / Redis / Mooncake Store
```

---

## 8. Prefill-Decode (PD) Disaggregation

### Motivation

Prefill and decode have fundamentally different performance profiles:
- **Prefill** is compute-bound — processes the entire input context in parallel
- **Decode** is memory-bandwidth-bound — generates one token at a time

Running them on the same GPU forces a compromise that is optimal for neither. Disaggregating them allows independent scaling and tighter latency control — both TTFT (Time to First Token) and ITL (Inter-Token Latency).

### Architecture

```
N × vLLM Prefill Instances    M × vLLM Decode Instances
         ↓                              ↑
    [KV Transfer Layer: LMCache / Mooncake / NIXL]
```

Autoscale N and M independently based on the live request mix. When a prefill completes, it writes KV blocks to the shared transfer layer; the decode instance picks them up.

### The KV Transfer Bottleneck

With a default block size of 16 tokens, naive block-by-block KV transfer results in many tiny transfers that underutilize network bandwidth. LMCache solves this by batching KV blocks into a single large contiguous buffer before transmission — analogous to how operating systems use I/O buffers to amortize syscall and network overhead.

---

## 9. Mooncake: KVCache-Centric Architecture

Mooncake is the serving platform for Kimi (Moonshot AI), open-sourced and winner of the **Best Paper Award at FAST 2025**. It represents the most complete production implementation of KVCache-centric disaggregation.

### Core Components

**Transfer Engine:** A high-performance data transport layer supporting:
- RDMA (InfiniBand, RoCEv2, GPUDirect RDMA)
- TCP fallback
- CUDA IPC (intra-node zero-copy)

**Mooncake Store:** A distributed KV cache storage pool built on Transfer Engine. Key properties:
- Multiple data replicas per object to alleviate access hotspots
- Striping and parallel I/O across multiple NICs (aggregated bandwidth)
- KV block addressing by prefix hash (compatible with vLLM's APC scheme)

**Conductor:** The global request scheduler. Conductor dispatches requests based on the current distribution of KVCache and workloads. It also proactively replicates or migrates hot KV blocks for future reuse.

### Request Flow

```
1. Conductor receives request
2. Identifies reusable prefix blocks in Mooncake Store (by hash)
3. Prefill node receives: raw input + reusable block IDs
4. Incremental prefill: only compute cache-miss tokens
5. New KV blocks written to Mooncake Store
6. KV transferred (RDMA) to decode node
7. Decode node streams tokens
```

This is the **xPyD** model: X prefill nodes and Y decode nodes sharing a distributed KV store.

### vLLM Integration

Mooncake plugs into vLLM via the **KV Connector API** (standardized plugin interface):

```bash
# Prefill node
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_producer"}'

# Decode node
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_consumer"}'
```

Mooncake Transfer Engine has been directly integrated into vLLM v1 as a first-class KV Connector (December 2024 / December 2025).

### Production Scale: Kimi K2

Mooncake powers Kimi K2 deployment on 128 H200 GPUs with PD disaggregation and large-scale expert parallelism:

| Metric | Value |
|---|---|
| Prefill throughput | 224,000 tokens/sec |
| Decode throughput | 288,000 tokens/sec |
| Weight update time (1T params) | ~21–22 seconds (pipelined H2D → Broadcast → Reload) |

### Mooncake vs. LMCache: Complementary Layers

These two are not competitors — they operate at different layers:

| Layer | Component | Role |
|---|---|---|
| Inference engine | vLLM | Computation, PagedAttention, APC |
| Caching logic | LMCache | What to cache, eviction, PD proxy |
| Storage backend | Mooncake Store | Distributed DRAM/SSD pool |
| Transport | Transfer Engine | RDMA / GPUDirect data movement |

---

## 10. KV-Cache Aware Routing at Scale

### The Problem with Naive Load Balancing

Single-instance prefix caching breaks down at cluster scale. Each vLLM pod manages its own GPU cache in isolation. Standard load balancers using cache-blind metrics (round-robin, least connections) scatter related requests across different pods, destroying cache locality and forcing redundant prefill computation.

### KV-Cache Aware Routing (llm-d / Red Hat)

**llm-d** implements state-aware request scheduling with a **KV cache indexer** that maintains a global, near-real-time view of KV block locality across all vLLM pods.

**Write path (cache events):**
1. vLLM pods publish `BlockStored` / `BlockRemoved` events via ZMQ
2. Events are topic-formatted as `kv@pod-id@model` and sharded by pod ID for ordered processing
3. Events are decoded from msgpack payloads
4. The indexer updates its global prefix tree

**Routing path:**
1. New request arrives at the gateway
2. Router hashes the request prefix block-by-block
3. Router queries the KV indexer: which pod has the longest matching prefix cached?
4. Request is forwarded to the optimal pod

**Performance impact:** KV-cache aware scheduling delivers:
- **57× faster response times** on prefix-heavy workloads
- **2× throughput** on identical hardware

---

## 11. Full Production Stack

### Complete Architecture Diagram

```
Request arrives
      ↓
[KV-Aware Router]
  llm-d / NVIDIA Dynamo / custom gateway
  Scores pods by: prefix hash match count in KV indexer
      ↓
[vLLM Prefill Pod]
  - PagedAttention block allocation
  - APC: hash match → GPU blocks reused
  - Incremental prefill for cache-miss tokens
      ↓                              ↑
[KV Transfer Layer]          [vLLM Decode Pod]
  LMCache connector API         - Reads KV from local GPU
  Mooncake Store backend          or Mooncake Store
  Transfer Engine (RDMA)        - Streams output tokens
      ↓
[Storage Tier]
  CPU DRAM (fast, volatile)
  Local NVMe (persistent, single-node)
  Remote nodes via RDMA (cluster-wide)
  Ceph / Alluxio (object storage for very long contexts)
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| **PagedAttention** | Non-contiguous GPU memory allocation for KV blocks |
| **APC (vLLM built-in)** | Hash-based prefix block reuse within a single pod |
| **LMCache** | Cross-session caching, eviction policy, PD proxy logic |
| **Mooncake Store** | Distributed DRAM/SSD pool with RDMA transfer |
| **Mooncake Transfer Engine** | High-throughput RDMA/GPUDirect KV data movement |
| **Conductor (Mooncake)** | Global KV-aware request scheduling |
| **llm-d / Dynamo** | Cluster-level KV-aware routing |
| **Ray** | Worker orchestration, placement groups, RLHF weight sync |

---

## 12. Practical Recommendations

### For Single-Node Deployment

- Use the default multiprocessing executor (`mp`) — it outperforms Ray for single-node.
- Enable APC (`--enable-prefix-caching`) — zero cost, immediate wins for any repeated prompt patterns.
- Structure prompts to maximize shared prefixes: system prompt + tool schemas first, conversation history after.

### For Multi-Node Deployment

- Use Ray executor with explicit `--distributed-executor-backend ray`.
- Set `tensor_parallel_size` = GPUs per node, `pipeline_parallel_size` = number of nodes.
- For serious scale, adopt PD disaggregation with LMCache or Mooncake as the KV transfer backend.

### For Agentic Workloads

| Concern | Recommendation |
|---|---|
| Multi-turn conversation | Enable APC — biggest single win |
| Long tool call histories | Keep tool schema + system prompt at the front (stable prefix head) |
| Context management | **Never truncate the prefix** — context truncation halves cache hit rate |
| Distributed cluster (>4 pods) | Use KV-cache aware routing (llm-d, NVIDIA Dynamo) |
| Very long contexts (>128K tokens) | LMCache + Mooncake Store or Ceph for cold tier storage |
| TTFT vs. throughput tradeoff | Enable PD disaggregation |

### For RLHF / Post-Training

| Need | Recommendation |
|---|---|
| Simplicity + flexibility | OpenRLHF (separate resource pools, NCCL/IPC weight sync) |
| Maximum GPU efficiency | veRL (co-located HybridEngine, in-memory resharding) |
| Simple prototyping | TRL with vLLM server mode (HTTP, no Ray required) |
| Very large scale (>100 GPUs) | Mooncake Checkpoint-Engine for weight synchronization |

### Avoiding Common Pitfalls

- **Ray Serve + TP > 1:** Use `distributed_executor_backend="mp"` to avoid nested placement group conflicts.
- **CPU backend:** Ray is not supported and falls back to multiprocessing automatically — this is by design.
- **Context truncation:** Avoid strategies that modify the stable prefix head of the context, as they destroy prefix cache hit rates.
- **Block size tuning:** The default 16-token block size is deliberately small for cache granularity. Larger blocks improve compute efficiency but reduce APC hit rates.

---

*Report generated from a live technical deep-dive on vLLM source code, Ray integration internals, RLHF framework architectures (veRL, OpenRLHF), and modern KV cache management systems (LMCache, Mooncake, llm-d).*
