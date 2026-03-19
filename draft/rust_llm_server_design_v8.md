# Rust LLM Inference Server — Design & Implementation Plan (v8)

> **Target:** 1–N machines with CXL-interconnected memory pools, 8–16 GPU/NPU cards per machine, hardware-limit inference for agent/coding workloads
> **Philosophy:** Rust for orchestration and scheduling. Reuse CUDA/NPU kernels via FFI. Exploit CXL/PGAS to eliminate data copies. Never trust shared memory for control structures. **The Linqu hierarchy is a structural principle, not a mapping table: every level instantiates the same parameterized abstractions.**
> **New in v8:** Recursive decomposition inspired by the Linqu distributed runtime. Unified `LevelNode`/`BlockStore`/`LevelAllocator` trait hierarchy, recursive routing, level-parameterized block pools, hierarchical gossip, uniform integrity and failure handling, cascading eviction. All level-specific components from v4–v7 are preserved as implementations behind uniform traits.

---

## Table of Contents

1. [Critique of v7 and What v8 Fixes](#1-critique-of-v7-and-what-v8-fixes)
2. [Linqu Recursive Decomposition Principles for LLM Serving](#2-linqu-recursive-decomposition-principles-for-llm-serving)
3. [Core Abstractions: Traits and Types](#3-core-abstractions-traits-and-types)
4. [The Recursive Hierarchy Tree](#4-the-recursive-hierarchy-tree)
5. [Recursive Routing](#5-recursive-routing)
6. [Level-Parameterized Block Pools](#6-level-parameterized-block-pools)
7. [Hierarchical Gossip and Discovery](#7-hierarchical-gossip-and-discovery)
8. [Uniform Integrity Model](#8-uniform-integrity-model)
9. [Recursive Failure Handling](#9-recursive-failure-handling)
10. [Hierarchical Capacity Management and Eviction](#10-hierarchical-capacity-management-and-eviction)
11. [What Changes from v7](#11-what-changes-from-v7)
12. [Risks and Mitigations](#12-risks-and-mitigations)
13. [Configuration](#13-configuration)
14. [Updated Directory Layout](#14-updated-directory-layout)
15. [Implementation Phases](#15-implementation-phases)
16. [Performance Targets](#16-performance-targets)
17. [Mapping to Linqu Hierarchy — Structural, Not Tabular](#17-mapping-to-linqu-hierarchy--structural-not-tabular)
18. [Meta Index Durability and Cost at TB-Scale](#18-meta-index-durability-and-cost-at-tb-scale)
19. [Layer-Wise KVCache Management and Compute-IO Overlap](#19-layer-wise-kvcache-management-and-compute-io-overlap)

---

## 1. Critique of v7 and What v8 Fixes

### Critique

v7 achieves corruption-resilient CXL caching — a major reliability win. But its architecture is **level-specific**, not **level-parameterized**. Every hierarchy level has its own bespoke implementation for the same logical operations. This works, but it violates the Linqu principle of hierarchical symmetry and creates several practical problems:

| # | v7 Problem | Why It Matters |
|---|---|---|
| 1 | **Four separate KV cache implementations.** `RadixKvManager` at L2 (GPU), `LocalBlockIndex` at L3 (host DRAM), `IndexGossip` at L4–L5 (CXL fabric), `RemoteStoreConnector` at L6 (cross-fabric). Each with its own API surface, error types, and configuration schema. | Adding a new level (e.g., NVMe tier, disaggregated GPU pool) requires writing a completely new implementation from scratch. Testing is per-implementation, not per-interface. Bugs in one level's logic are not caught by another level's tests. |
| 2 | **Flat routing.** `CxlAwareRouter` scores ALL instances globally with a single distance-weighted formula. It has no concept of "pick the best switch domain, then pick the best host within that domain." | At 8+ hosts across 2+ CXL switches, flat routing evaluates O(N) instances per request. Hierarchical routing evaluates O(log N) decisions, each at the appropriate level's granularity. More importantly, flat routing cannot exploit topology-aware locality — it cannot prefer "same switch domain" over "same fabric but different switch." |
| 3 | **Flat gossip.** `IndexGossip` broadcasts to all peers uniformly. Every host sends its entire delta to every other host. | At 8 hosts, gossip bandwidth is `8 × 8 × delta_size` per interval. With hierarchical gossip (L3 reports to L4 leader, L4 leader summarizes for L5), bandwidth scales as `8 × delta_size` (each host reports once to its leader). For small clusters this doesn't matter; for 64+ hosts it does. |
| 4 | **Level-specific allocation.** GPU has its own `BlockPool`, CXL has `CxlMemoryManager`, and L6 has no allocator at all. There is no uniform way to ask "how many blocks can level X hold?" or "allocate N blocks at level X." | The pool manager (`CxlAwarePoolManager`) has level-specific code paths for GPU eviction, CXL eviction, and remote store interaction. Adding a tier means modifying the pool manager — it cannot be extended by adding a new `LevelNode`. |
| 5 | **Level-specific failure handling.** GPU health checks, CXL corruption detection (checksum + quarantine), and remote store failure (RPC timeout) are all separate code paths with no shared pattern. | The detect-quarantine-recover pattern is the same at every level but implemented differently each time. A bug in the quarantine logic at one level may not be caught at another. |
| 6 | **Level-specific metrics.** `cxl_promote_ok`, `cxl_corruption_detected`, `gpu_blocks_free`, `remote_store_latency_ms` — each level has bespoke metric names. | You cannot compare levels side-by-side in a dashboard. You cannot write a single alert rule that says "if any level's miss rate exceeds X%." Each level requires its own monitoring configuration. |
| 7 | **The "Mapping to Linqu Hierarchy" section is a table, not a structure.** §18 says "this component maps to that level" but the code does not enforce or exploit this mapping. The hierarchy is descriptive, not prescriptive. | The Linqu runtime achieves structural symmetry: `task_ring[L][d]` exists at every level with the same API, just different capacity. The LLM server could achieve the same: `block_pool[L]` at every level with the same API, just different backing store. This would make the hierarchy a design constraint that guides implementation, not just a documentation artifact. |

### The Structural Principle (from Linqu)

The Linqu distributed runtime (see `linqu_runtime_design.md`) defines a 7-level hierarchy where **every runtime component is structured around the same hierarchy**:

- `task_ring[L][d]` and `buffer_ring[L][d]` — the SAME data structure at every level, parameterized by level `L` and scope depth `d`.
- `pl.at(level=...)` — the SAME programming interface at every level.
- Orchestrator/Worker roles — the SAME thread model at every level.
- Scope-exit semantics — the SAME retirement rules at every level.

v7's LLM server has the **levels** but not the **symmetry**. v8 adds the symmetry.

### What v8 Adds

v8 retains all of v7's corruption-resilient CXL architecture and adds a recursive decomposition layer:

- **§3**: Core traits (`LevelNode`, `BlockStore`, `LevelAllocator`, `IntegrityVerifier`) that every level implements
- **§4**: A recursive hierarchy tree built at startup from CXL topology discovery
- **§5**: Recursive routing — each level picks the best child, not a flat global score
- **§6**: Level-parameterized block pools — `block_pool[L]` with uniform API
- **§7**: Hierarchical gossip — same protocol, scoped by level
- **§8**: Uniform integrity model — same `verify()` interface, level-specific implementation
- **§9**: Recursive failure handling — detect/quarantine/recover at every level
- **§10**: Hierarchical capacity management — cascading eviction up the hierarchy

### What v8 Does NOT Change

- CXL block I/O (`CxlBlockHeader`, `cxl_write_block`, `cxl_read_block`) — unchanged, now the L3–L5 `BlockStore` internals
- UDP heartbeat mechanism — unchanged, now the L3–L5 health check implementation
- CXL topology discovery — unchanged, now the input to tree construction
- Allocator standby failover — unchanged, now the L4 `LevelAllocator` implementation detail
- CXL scrubber — unchanged, now the L3–L5 `IntegrityVerifier::scrub()` implementation
- All v4 components (HAL, scheduler, forward pass, CUDA graphs, detokenizer, metrics HTTP API)
- Shared weight pool (orthogonal to KV block hierarchy)
- Control plane / data plane separation (now an implementation detail of L3–L5, not a top-level concern)

---

## 2. Linqu Recursive Decomposition Principles for LLM Serving

### 2.1 Five Principles That Transfer

The Linqu runtime (`linqu_runtime_design.md` §1–§2) defines principles that directly apply to LLM KV cache management:

| Linqu Principle | LLM Server Application |
|---|---|
| **Hierarchical symmetry**: every runtime component mirrors the physical hierarchy. | Every KV cache level (GPU, Host DRAM, CXL switch, CXL fabric, remote) implements the same `BlockStore` trait. There is no separate "software topology" — the trait hierarchy IS the hardware hierarchy. |
| **Recursive enclosure**: Level N encloses several Level N-1 instances. | An L5 `FabricNode` encloses several L4 `SwitchDomainNode`s, each of which encloses several L3 `HostNode`s, each of which encloses several L2 `ChipNode`s. |
| **Level-parameterized data structures**: `ring[L][d]` at every level, same structure. | `block_pool[L]` at every level: same `BlockStore` trait, different backing store (HBM, DRAM, CXL, RDMA). |
| **Unified interface**: `pl.at(level=...)` works at every level. | `LevelNode::route()`, `LevelNode::store()`, `LevelNode::evict()` work at every level. |
| **Three-tier communication**: shared memory (L0–L2), DMA (L2–L3), message passing (L3–L6). | GPU DMA (L2–L3), CXL load/store (L3–L5), RPC/RDMA (L5–L6). Different transport, same abstract `BlockStore::fetch`/`store` interface. |

### 2.2 What Does NOT Transfer

Some Linqu concepts are specific to its task-parallel execution model and do not apply to LLM serving:

| Linqu Concept | Why It Doesn't Apply |
|---|---|
| **Ring-buffer task scheduling** (`task_ring[L][d]`) | LLM server uses request-level scheduling (continuous batching), not scope-based task rings. |
| **SPMD fan-out** (`pl.at(level=CLUSTER_0)` dispatches to all nodes) | LLM server routes individual requests to specific instances, not SPMD programs. |
| **Scope depth** (`d` in `buffer_ring[L][d]`) | KV blocks have flat lifetime (hash-based identity, LRU eviction), not nested scope lifetime. |
| **Producer/consumer task keys** (`TaskKey(scope_level, task_id)`) | KV blocks are identified by `BlockHash` (SHA-256 of token prefix), not task coordinates. |
| **`pl.free(tensor)` early scope release** | KV blocks use reference counting and LRU, not scope tokens. |

### 2.3 The Hierarchy for LLM Serving

```
Level:  L2 (Chip)     L3 (Host)       L4 (SwitchDomain)    L5 (Fabric)      L6 (Global)
Store:  GPU HBM       Host DRAM +     CXL pooled memory    CXL fabric       RDMA/TCP
                      CXL-local       (same switch)        (cross switch)   remote store
Lat:    ~1–2ns        ~80–200ns       ~250–350ns           ~400–600ns       ~1–50μs
Prot:   ECC (HW)      ECC + xxHash    xxHash (CXL shared)  xxHash           TCP checksum
```

The same `BlockStore` interface provides `contains`/`fetch`/`store`/`remove` at every level. The same `LevelNode` interface provides `route`/`evict`/`health`/`verify` at every level. The implementation behind the trait varies — that is the point.

---

## 3. Core Abstractions: Traits and Types

### 3.1 `HierarchyLevel` Enum

```rust
/// The Linqu-aligned hierarchy level.
/// Maps directly to the Linqu machine hierarchy (Levels 0–6),
/// restricted to the levels relevant to KV cache management.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum HierarchyLevel {
    /// Level 2: Chip (GPU/NPU). KV blocks in HBM.
    Chip = 2,
    /// Level 3: Host. KV blocks in host DRAM or CXL-attached expander.
    Host = 3,
    /// Level 4: Cluster-level-0. KV blocks in CXL-pooled memory (same switch).
    SwitchDomain = 4,
    /// Level 5: Cluster-level-1. KV blocks across CXL switches (same fabric).
    Fabric = 5,
    /// Level 6: Cluster-level-2. KV blocks via RDMA/TCP (cross-fabric).
    Global = 6,
}

impl HierarchyLevel {
    /// The child level (one step down in the hierarchy).
    pub fn child(&self) -> Option<HierarchyLevel> {
        match self {
            Self::Global => Some(Self::Fabric),
            Self::Fabric => Some(Self::SwitchDomain),
            Self::SwitchDomain => Some(Self::Host),
            Self::Host => Some(Self::Chip),
            Self::Chip => None,
        }
    }

    /// The parent level (one step up in the hierarchy).
    pub fn parent(&self) -> Option<HierarchyLevel> {
        match self {
            Self::Chip => Some(Self::Host),
            Self::Host => Some(Self::SwitchDomain),
            Self::SwitchDomain => Some(Self::Fabric),
            Self::Fabric => Some(Self::Global),
            Self::Global => None,
        }
    }

    /// Linqu-compatible numeric level.
    pub fn linqu_level(&self) -> u8 {
        *self as u8
    }
}
```

### 3.2 `BlockStore` Trait — Unified Storage Interface

This trait replaces v5's `KvConnector` trait. Every hierarchy level implements it. The key change from `KvConnector`: it includes `level()` as a first-class part of the interface and adds `capacity`/`usage` methods for uniform capacity management.

```rust
/// Universal block identifier: SHA-256 hash of the token prefix.
/// Same as v4–v7. Unchanged.
pub type BlockHash = [u8; 32];

/// A handle to a block at a specific hierarchy level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockHandle {
    /// Which level this block lives at.
    pub level: HierarchyLevel,
    /// Level-specific address. Interpretation varies:
    /// - L2 (Chip): GPU block ID (u32 cast to u64)
    /// - L3–L5 (Host/CXL): GlobalBlockAddr encoded as (region_id << 32 | offset)
    /// - L6 (Global): remote store key
    pub addr: u64,
    /// Block hash (for integrity verification).
    pub hash: BlockHash,
}

/// The unified storage interface. Every hierarchy level implements this.
///
/// This is the recursive analog of Linqu's `buffer_ring[L][d]` — the same
/// data structure at every level, with level-specific backing store.
pub trait BlockStore: Send + Sync {
    /// Which hierarchy level this store operates at.
    fn level(&self) -> HierarchyLevel;

    /// Check which blocks exist in this store. Batch API for efficiency.
    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool>;

    /// Fetch block data from this store. Returns None for blocks not found.
    /// All reads are integrity-verified (the implementation decides how).
    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>>;

    /// Store blocks into this level. Integrity metadata (checksums) are
    /// computed and stored by the implementation.
    fn store(&self, blocks: &[KvBlockData]) -> Vec<Result<BlockHandle, BlockError>>;

    /// Remove blocks from this store.
    fn remove(&self, hashes: &[BlockHash]);

    /// Total capacity in blocks at this level.
    fn capacity_blocks(&self) -> u32;

    /// Current usage in blocks at this level.
    fn used_blocks(&self) -> u32;

    /// Level name for logging/metrics.
    fn level_name(&self) -> &str;
}

/// Block data with metadata. Same as v5 KvBlockData, unchanged.
pub struct KvBlockData {
    pub meta: KvBlockMeta,
    pub data: Vec<u8>,
}

/// Errors that any BlockStore can produce. Uniform across levels.
#[derive(Debug)]
pub enum BlockError {
    /// No space available at this level.
    CapacityExhausted,
    /// Integrity verification failed (checksum, ECC, etc.).
    IntegrityFailure { hash: BlockHash, detail: String },
    /// Block not found.
    NotFound { hash: BlockHash },
    /// Transport/IO error (RPC timeout, DMA failure, etc.).
    TransportError(String),
}
```

**Implementations:**

| Level | Implementing Struct | Backing Store | Notes |
|---|---|---|---|
| L2 | `GpuBlockStore` | GPU HBM via `BlockPool` | Wraps v4's `RadixKvManager` + `BlockPool`. The radix tree is an internal optimization invisible to the `BlockStore` trait. |
| L3 | `HostBlockStore` | Host DRAM + CXL-local expander | Wraps v7's `LocalBlockIndex` + `SymmetricHeap`. `fetch()` calls `cxl_read_block` with checksum verification. `store()` calls `cxl_write_block`. |
| L4 | `SwitchDomainBlockStore` | CXL-pooled memory (same switch) | Same CXL backing as L3 but for blocks in the pooled region. Uses `CxlMemoryManager` for allocation. |
| L5 | `FabricBlockStore` | CXL fabric memory (cross-switch) | Same CXL backing as L4 but at higher latency. |
| L6 | `GlobalBlockStore` | RDMA/TCP remote store | Wraps v5's `RemoteStoreConnector`. |

### 3.3 `LevelNode` Trait — The Recursive Hierarchy Node

This is the structural backbone of v8. Each `LevelNode` represents one node in the hierarchy tree. It has children (next-level-down nodes) and provides a uniform interface for routing, eviction, health checking, and gossip.

```rust
/// A node in the recursive hierarchy tree.
///
/// This is the LLM-serving analog of Linqu's recursive enclosure model:
/// "Each level is a logical machine that encloses several instances of
/// the level below" (linqu_runtime_design.md §2.2).
///
/// The key recursive property: `children()` returns `LevelNode`s at the
/// next level down. An L5 FabricNode's children are L4 SwitchDomainNodes.
/// An L4 SwitchDomainNode's children are L3 HostNodes. And so on.
pub trait LevelNode: Send + Sync {
    /// This node's hierarchy level.
    fn level(&self) -> HierarchyLevel;

    /// This node's local block store.
    fn store(&self) -> &dyn BlockStore;

    /// Child nodes (next level down in the hierarchy).
    /// Empty for L2 (Chip) — the leaf level.
    fn children(&self) -> &[Arc<dyn LevelNode>];

    /// Select the best child for a set of block hashes.
    /// Returns the index into `children()`.
    ///
    /// This is the recursive routing operation: each level picks the
    /// best child using level-appropriate criteria (CXL distance at L4,
    /// cache hit rate at L3, GPU block availability at L2).
    fn route(&self, hashes: &[BlockHash], load: &LoadSnapshot) -> usize;

    /// Evict blocks from this level's store. Returns the evicted blocks
    /// (hash + data) so the caller can push them to the parent level.
    ///
    /// This is the cascading eviction operation: L2 evicts to L3,
    /// L3 evicts to L4, L4 evicts to L5, L5 evicts to L6.
    fn evict(&self, count: u32) -> Vec<(BlockHash, Vec<u8>)>;

    /// Verify integrity of a block at this level.
    fn verify(&self, handle: &BlockHandle) -> Result<(), IntegrityError>;

    /// Health status of this node and its children.
    fn health(&self) -> LevelHealth;

    /// Summary of this node's state (for hierarchical gossip).
    /// The parent level aggregates children's summaries.
    fn summary(&self) -> LevelSummary;

    /// Uniform metrics for this level.
    fn metrics(&self) -> &LevelMetrics;
}

/// Load information for routing decisions.
pub struct LoadSnapshot {
    /// Per-child: number of active sequences.
    pub active_sequences: Vec<u32>,
    /// Per-child: GPU/device utilization (0.0–1.0).
    pub utilization: Vec<f32>,
    /// Per-child: available KV block capacity.
    pub free_blocks: Vec<u32>,
}

/// Health status — uniform across levels.
#[derive(Clone, Debug)]
pub struct LevelHealth {
    pub level: HierarchyLevel,
    pub status: HealthStatus,
    /// Number of healthy children.
    pub healthy_children: u32,
    /// Number of degraded/failed children.
    pub unhealthy_children: u32,
    /// Integrity failures detected since last reset.
    pub integrity_failures: u64,
    /// Quarantined blocks at this level.
    pub quarantined_blocks: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HealthStatus {
    /// All children healthy, no integrity failures.
    Healthy,
    /// Some children unhealthy or some integrity failures, but operational.
    Degraded,
    /// Node is not operational.
    Failed,
}

/// Summary of a node's block index — for hierarchical gossip.
/// Each level produces this; the parent aggregates children's summaries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelSummary {
    pub level: HierarchyLevel,
    pub node_id: u32,
    /// Block hashes stored at this level (or a bloom filter for large sets).
    pub block_hashes: Vec<BlockHash>,
    /// Total blocks stored.
    pub block_count: u32,
    /// Free capacity.
    pub free_blocks: u32,
    /// Timestamp for freshness.
    pub timestamp_ms: u64,
}

/// Uniform metrics — same schema at every level.
/// Replaces v7's ad-hoc per-component metrics.
pub struct LevelMetrics {
    pub level: HierarchyLevel,
    pub lookups: AtomicU64,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub stores: AtomicU64,
    pub evictions: AtomicU64,
    pub promotions: AtomicU64,    // blocks promoted FROM this level to a lower level
    pub demotions: AtomicU64,     // blocks demoted TO this level from a lower level
    pub integrity_checks: AtomicU64,
    pub integrity_failures: AtomicU64,
    pub quarantined: AtomicU64,
    pub fetch_latency_ns: AtomicU64,  // moving average
    pub store_latency_ns: AtomicU64,  // moving average
}

impl LevelMetrics {
    pub fn new(level: HierarchyLevel) -> Self {
        Self {
            level,
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            stores: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
            demotions: AtomicU64::new(0),
            integrity_checks: AtomicU64::new(0),
            integrity_failures: AtomicU64::new(0),
            quarantined: AtomicU64::new(0),
            fetch_latency_ns: AtomicU64::new(0),
            store_latency_ns: AtomicU64::new(0),
        }
    }

    /// Hit rate at this level.
    pub fn hit_rate(&self) -> f64 {
        let total = self.lookups.load(Ordering::Relaxed);
        if total == 0 { return 0.0; }
        self.hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Occupancy (used / capacity).
    pub fn occupancy(&self, store: &dyn BlockStore) -> f64 {
        let cap = store.capacity_blocks();
        if cap == 0 { return 0.0; }
        store.used_blocks() as f64 / cap as f64
    }
}
```

### 3.4 `LevelAllocator` Trait

```rust
/// Block allocation at a specific hierarchy level.
///
/// Separated from `BlockStore` because allocation ownership may differ
/// from storage access. At L2, the GPU owns allocation. At L3–L5, the
/// centralized CxlMemoryManager (v7 §6) owns allocation. At L6, the
/// remote store owns allocation.
pub trait LevelAllocator: Send + Sync {
    fn level(&self) -> HierarchyLevel;
    fn allocate(&self, count: u32) -> Result<Vec<BlockHandle>, AllocError>;
    fn free(&self, handles: &[BlockHandle]);
    fn quarantine(&self, handle: BlockHandle);
    fn free_count(&self) -> u32;
    fn total_count(&self) -> u32;
}

#[derive(Debug)]
pub enum AllocError {
    /// No free blocks available.
    Exhausted,
    /// Allocator is not the primary (for centralized allocators).
    NotPrimary,
    /// Communication error with allocator.
    RpcError(String),
}
```

**Implementations:**

| Level | Implementing Struct | Notes |
|---|---|---|
| L2 | `GpuAllocator` | Wraps v4's `BlockPool`. Allocation is local (GPU memory). |
| L3–L5 | `CxlAllocatorClient` | Wraps v7's `AllocatorClient`. Sends RPC to centralized `CxlMemoryManager`. The `CxlMemoryManager` itself (v7 §6) is unchanged — it becomes the server behind the `LevelAllocator` trait for L3–L5. |
| L6 | `RemoteAllocator` | Delegates to remote store's allocation API. |

### 3.5 `LevelConfig` — Uniform Per-Level Configuration

```rust
/// Configuration for one hierarchy level.
/// Same schema at every level; different defaults.
///
/// This is the LLM-serving analog of Linqu's level-parameterized approach:
/// the SAME configuration structure at every level, with level-specific
/// defaults and values.
#[derive(Clone, Debug, Deserialize)]
pub struct LevelConfig {
    pub level: HierarchyLevel,
    pub enabled: bool,
    pub capacity_blocks: u32,
    pub block_size_bytes: usize,
    pub integrity: IntegrityConfig,
    pub gossip: Option<GossipConfig>,
    pub health_check: HealthCheckConfig,
    pub eviction: EvictionConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IntegrityConfig {
    /// Checksum algorithm. Level-specific defaults:
    /// L2: None (ECC handles it), L3–L5: XxHash64, L6: None (TCP handles it).
    pub checksum: ChecksumAlgo,
    /// Verify checksum on every read.
    pub verify_on_read: bool,
    /// Background scrubbing.
    pub scrub_enabled: bool,
    pub scrub_interval_secs: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub enum ChecksumAlgo {
    /// No software checksum (hardware ECC or transport checksum suffices).
    None,
    /// xxHash64 (~30 GB/s, 64-bit). Used for CXL blocks.
    XxHash64,
    /// CRC32c (HW-accelerated on x86). Alternative to xxHash64.
    Crc32c,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GossipConfig {
    pub listen_addr: String,
    pub peers: Vec<String>,
    pub interval_ms: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HealthCheckConfig {
    pub interval_ms: u64,
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EvictionConfig {
    /// Start evicting when occupancy exceeds this threshold.
    pub high_watermark: f64,   // e.g., 0.90
    /// Stop evicting when occupancy drops below this threshold.
    pub low_watermark: f64,    // e.g., 0.70
    /// Eviction policy.
    pub policy: EvictionPolicy,
}

#[derive(Clone, Debug, Deserialize)]
pub enum EvictionPolicy {
    /// Least recently used.
    Lru,
    /// Longest unused prefix (v4's radix tree eviction at L2).
    LongestUnusedPrefix,
    /// No eviction (discard when full — used at L6 fallback).
    Discard,
}
```

### 3.6 `IntegrityVerifier` Trait

```rust
/// Integrity verification at a specific hierarchy level.
///
/// The interface is the same at every level. The implementation varies:
/// - L2: ECC (hardware, no software overhead). `verify()` is a no-op.
/// - L3–L5: xxHash64 checksum. `verify()` reads the block and checks
///   the checksum against the header (v7 §4.4).
/// - L6: TCP/transport checksum. `verify()` is a no-op (transport handles it).
pub trait IntegrityVerifier: Send + Sync {
    fn level(&self) -> HierarchyLevel;

    /// Verify a block's integrity. Returns Ok if the block is valid.
    fn verify(&self, handle: &BlockHandle) -> Result<(), IntegrityError>;

    /// Background scrub: verify all allocated blocks.
    /// Returns the number of corrupted blocks found.
    fn scrub(&self) -> u32;

    /// Quarantine a block (remove from use, do not free for reuse).
    fn quarantine(&self, handle: &BlockHandle);

    /// Number of quarantined blocks.
    fn quarantined_count(&self) -> u32;
}

#[derive(Debug)]
pub struct IntegrityError {
    pub level: HierarchyLevel,
    pub handle: BlockHandle,
    pub kind: IntegrityErrorKind,
}

#[derive(Debug)]
pub enum IntegrityErrorKind {
    /// Block header is invalid (uninitialized or gross corruption).
    InvalidHeader,
    /// Block hash mismatch (index pointed to wrong block).
    HashMismatch { expected: BlockHash, got: BlockHash },
    /// Data checksum failed (bit-level corruption).
    ChecksumMismatch { expected: u64, actual: u64 },
    /// Hardware-reported error (ECC uncorrectable, PCIe error).
    HardwareError(String),
}
```

**Implementations:**

| Level | Struct | Notes |
|---|---|---|
| L2 | `GpuIntegrityVerifier` | `verify()` is a no-op (ECC). `scrub()` reads all GPU blocks and checks for ECC errors (optional). `quarantine()` marks a GPU block as unusable. |
| L3–L5 | `CxlIntegrityVerifier` | Wraps v7's `CxlScrubber`. `verify()` calls `cxl_read_block` with checksum. `scrub()` runs a full pass over all allocated CXL blocks. `quarantine()` calls `CxlMemoryManager::quarantine()`. |
| L6 | `RemoteIntegrityVerifier` | `verify()` is a no-op (TCP). `scrub()` is a no-op. `quarantine()` removes the remote key. |

---

## 4. The Recursive Hierarchy Tree

### 4.1 Runtime Tree Structure

At startup, the server constructs a hierarchy tree from CXL topology discovery (v6 §2, unchanged) and configuration:

```
                         ┌─────────────────────┐
                         │  GlobalNode (L6)     │
                         │  store: RemoteStore  │
                         └──────────┬──────────┘
                                    │ children
                    ┌───────────────┴───────────────┐
                    │                               │
           ┌────────┴────────┐             ┌────────┴────────┐
           │ FabricNode (L5) │             │ FabricNode (L5) │
           │ store: CXL xswi │             │ store: CXL xswi │
           └────────┬────────┘             └────────┬────────┘
                    │ children                      │ children
            ┌───────┴───────┐               ┌───────┴───────┐
            │               │               │               │
    ┌───────┴──────┐ ┌──────┴───────┐  ... (same pattern)
    │SwitchDomain  │ │SwitchDomain  │
    │Node (L4)     │ │Node (L4)     │
    │store: CXL sw │ │store: CXL sw │
    └───────┬──────┘ └──────┬───────┘
            │ children      │ children
      ┌─────┴─────┐   ┌────┴─────┐
      │           │   │          │
  ┌───┴───┐  ┌───┴───┐  ...
  │HostNode│  │HostNode│
  │(L3)    │  │(L3)    │
  │store:  │  │store:  │
  │DRAM+CXL│  │DRAM+CXL│
  └───┬────┘  └───┬────┘
      │children    │children
  ┌───┴───┐    ┌───┴───┐
  │ChipNode│   │ChipNode│
  │(L2)    │   │(L2)    │
  │store:  │   │store:  │
  │GPU HBM │   │GPU HBM │
  └────────┘   └────────┘
```

### 4.2 Level Collapsing for Simple Topologies

For a 2-host cluster with a single CXL switch, L4 = L5 = L6 collapses:

```
          ┌────────────────────┐
          │ GlobalNode (L6)    │
          │ (collapsed L4/L5)  │
          └─────────┬──────────┘
                    │ children
            ┌───────┴───────┐
            │               │
        ┌───┴───┐      ┌───┴───┐
        │HostNode│      │HostNode│
        │(L3)    │      │(L3)    │
        └───┬────┘      └───┬────┘
         ┌──┴──┐         ┌──┴──┐
      ┌──┴──┐┌─┴──┐   ┌──┴──┐┌─┴──┐
      │Chip0││Chip1│   │Chip0││Chip1│
      └─────┘└─────┘   └─────┘└─────┘
```

Level collapsing is determined at tree construction time. A level is collapsed when:
- There is only one node at that level (the hierarchy adds no fan-out), OR
- The level is not enabled in configuration (`LevelConfig::enabled = false`).

A collapsed level's `GlobalNode` directly wraps its children — no intermediate `FabricNode` or `SwitchDomainNode`. The `route()` call passes through without additional routing logic.

```rust
/// Build the hierarchy tree from topology and configuration.
pub fn build_hierarchy_tree(
    topology: &CxlTopology,
    config: &[LevelConfig],
    this_host: u32,
) -> Arc<dyn LevelNode> {
    // Step 1: Build L2 (Chip) leaf nodes from GPU devices.
    let chip_nodes: Vec<Arc<dyn LevelNode>> = enumerate_gpus()
        .map(|gpu| Arc::new(ChipNode::new(gpu)) as Arc<dyn LevelNode>)
        .collect();

    // Step 2: Build L3 (Host) node for this host, enclosing its L2 children.
    let host_node = Arc::new(HostNode::new(
        this_host,
        chip_nodes,
        config.iter().find(|c| c.level == HierarchyLevel::Host),
    ));

    // Step 3: Build L4–L6 from CXL topology, with level collapsing.
    let mut current_children: Vec<Arc<dyn LevelNode>> = vec![host_node];

    for level in [HierarchyLevel::SwitchDomain, HierarchyLevel::Fabric, HierarchyLevel::Global] {
        let level_config = config.iter().find(|c| c.level == level);
        let enabled = level_config.map(|c| c.enabled).unwrap_or(false);

        if enabled && should_create_level(level, topology) {
            // Group children by their parent at this level.
            let groups = group_children_by_topology(level, topology, &current_children);
            current_children = groups.into_iter()
                .map(|children| {
                    Arc::new(ClusterNode::new(level, children, level_config)) as Arc<dyn LevelNode>
                })
                .collect();
        }
        // If not enabled or single group, children pass through (level collapsed).
    }

    // The root is the topmost node.
    assert_eq!(current_children.len(), 1);
    current_children.into_iter().next().unwrap()
}
```

### 4.3 How the Tree Maps to v7's Existing Components

| Tree Node | v7 Component It Wraps |
|---|---|
| `ChipNode` (L2) | `RadixKvManager` + `BlockPool` (v4 §9) |
| `HostNode` (L3) | `LocalBlockIndex` + `SymmetricHeap` (v7 §5, §10) |
| `ClusterNode` (L4) | `CxlMemoryManager` + `IndexGossip` (v7 §6, §5.2) |
| `ClusterNode` (L5) | Same as L4, with cross-switch CXL regions |
| `ClusterNode` (L6) | `RemoteStoreConnector` (v5 §4.4) |

The tree does not replace these components — it wraps them behind the `LevelNode` and `BlockStore` traits, providing a uniform recursive interface while preserving the level-specific implementation details.

---

## 5. Recursive Routing

### 5.1 The Routing Algorithm

In v7, `CxlAwareRouter` scores all instances flat:

```
v7:  score(instance_i) = gpu_hit_weight × gpu_hits[i]
                       + cxl_hit_weight × cxl_hits[i] / distance_ns[i]
                       - load_penalty × active_sequences[i]
     route to argmax(score)
```

In v8, routing descends the hierarchy tree. Each level's `route()` picks the best child using level-appropriate criteria:

```
v8:  L6.route() → picks best L5 fabric domain
       L5.route() → picks best L4 switch domain within that fabric
         L4.route() → picks best L3 host within that switch domain
           L3.route() → picks best L2 GPU on that host
```

### 5.2 Per-Level Scoring

The scoring function is the SAME algorithm at every level, but with level-specific parameters:

```rust
/// Score a child node for routing a request with given block hashes.
/// Same algorithm at every level; only `distance_weight` and
/// `capacity_weight` vary.
fn score_child(
    child: &dyn LevelNode,
    hashes: &[BlockHash],
    load: &LoadSnapshot,
    child_idx: usize,
    params: &RoutingParams,
) -> f64 {
    let store = child.store();

    // How many of the requested blocks are present at this child?
    let contains = store.contains(hashes);
    let hit_count = contains.iter().filter(|&&b| b).count() as f64;
    let hit_fraction = hit_count / hashes.len().max(1) as f64;

    // Cache hit score: higher is better.
    let cache_score = hit_fraction * params.hit_weight;

    // Load score: lower utilization is better.
    let util = load.utilization.get(child_idx).copied().unwrap_or(0.0);
    let load_score = (1.0 - util as f64) * params.load_weight;

    // Capacity score: more free blocks is better.
    let free = load.free_blocks.get(child_idx).copied().unwrap_or(0) as f64;
    let cap = store.capacity_blocks().max(1) as f64;
    let capacity_score = (free / cap) * params.capacity_weight;

    cache_score + load_score + capacity_score
}
```

```rust
/// Per-level routing parameters. Same structure, different defaults.
pub struct RoutingParams {
    pub hit_weight: f64,
    pub load_weight: f64,
    pub capacity_weight: f64,
}

impl RoutingParams {
    /// Default parameters vary by level.
    pub fn defaults_for(level: HierarchyLevel) -> Self {
        match level {
            HierarchyLevel::Chip => Self {
                hit_weight: 10.0,    // GPU hits are very valuable (avoid recompute)
                load_weight: 1.0,
                capacity_weight: 0.5,
            },
            HierarchyLevel::Host => Self {
                hit_weight: 5.0,     // CXL hits save ~4μs vs recompute
                load_weight: 2.0,
                capacity_weight: 1.0,
            },
            HierarchyLevel::SwitchDomain | HierarchyLevel::Fabric => Self {
                hit_weight: 3.0,     // Cross-switch CXL hit saves less
                load_weight: 3.0,    // Load balancing matters more at cluster level
                capacity_weight: 1.0,
            },
            HierarchyLevel::Global => Self {
                hit_weight: 1.0,     // Remote hits have high latency anyway
                load_weight: 5.0,    // Prefer load balancing at top level
                capacity_weight: 2.0,
            },
        }
    }
}
```

### 5.3 Request Descent

A request enters at the root and descends:

```rust
/// Route a request through the hierarchy tree.
/// Returns the path from root to leaf (sequence of child indices).
pub fn route_request(
    root: &dyn LevelNode,
    hashes: &[BlockHash],
    load: &LoadSnapshot,
) -> Vec<usize> {
    let mut path = Vec::new();
    let mut current: &dyn LevelNode = root;

    loop {
        let children = current.children();
        if children.is_empty() {
            break;  // Reached leaf (L2)
        }

        let best_child = current.route(hashes, load);
        path.push(best_child);
        current = children[best_child].as_ref();
    }

    path
}
```

### 5.4 Comparison with v7 Flat Routing

| Property | v7 (Flat) | v8 (Recursive) |
|---|---|---|
| Complexity per request | O(N) where N = total instances | O(d × f) where d = tree depth, f = max fan-out per level |
| Topology awareness | Distance-weighted but flat | Inherent — each level routes within its topology scope |
| Adding a new level | Modify scoring formula | Add a `LevelNode` implementation; routing is automatic |
| Small cluster (2 hosts, 4 GPUs) | Scores 4 instances | Tree depth 3, fan-out 2: scores 2+2+2 = 6 children (slightly more overhead, but same quality) |
| Large cluster (64 hosts, 512 GPUs) | Scores 512 instances | Tree depth 5, fan-out 8: scores 8+8+8+8+8 = 40 children |

---

## 6. Level-Parameterized Block Pools

### 6.1 The Block Pool at Every Level

In Linqu: `buffer_ring[L][d]` — same ring structure at every level.
In v8: `block_pool[L]` — same `BlockStore` trait at every level.

Each level's `BlockStore` has the same operations (`contains`, `fetch`, `store`, `remove`, `capacity_blocks`, `used_blocks`) but different backing:

```
block_pool[L2] = GpuBlockStore     → GPU HBM blocks (v4 BlockPool)
block_pool[L3] = HostBlockStore    → Host DRAM HashMap + CXL-local (v7 LocalBlockIndex)
block_pool[L4] = CxlBlockStore     → CXL pooled memory, same switch (v7 SymmetricHeap)
block_pool[L5] = CxlBlockStore     → CXL fabric memory, cross switch (v7 SymmetricHeap)
block_pool[L6] = RemoteBlockStore  → RDMA/TCP remote store (v5 RemoteStoreConnector)
```

### 6.2 L2 Implementation: `GpuBlockStore`

Wraps v4's `RadixKvManager` and `BlockPool`:

```rust
/// L2 (Chip) block store. Wraps the GPU-local radix tree and block pool.
///
/// The radix tree is an internal optimization specific to L2 — it provides
/// prefix sharing (multiple sequences share common prefix blocks) and
/// tree-walk eviction. Higher levels use flat hash maps because prefix
/// relationships are only meaningful at the GPU level where the attention
/// kernel consumes them.
pub struct GpuBlockStore {
    /// The v4 radix tree + block pool. Unchanged internally.
    radix_manager: RwLock<RadixKvManager>,
    metrics: LevelMetrics,
}

impl BlockStore for GpuBlockStore {
    fn level(&self) -> HierarchyLevel { HierarchyLevel::Chip }

    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool> {
        let mgr = self.radix_manager.read();
        hashes.iter().map(|h| mgr.find_block_by_hash(h).is_some()).collect()
    }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        let mgr = self.radix_manager.read();
        hashes.iter().map(|h| {
            mgr.find_block_by_hash(h).map(|block_id| {
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
                KvBlockData {
                    meta: mgr.block_meta_by_hash(h),
                    data: mgr.gpu_pool.read_block(block_id),
                }
            })
        }).collect()
    }

    fn store(&self, blocks: &[KvBlockData]) -> Vec<Result<BlockHandle, BlockError>> {
        let mut mgr = self.radix_manager.write();
        blocks.iter().map(|block| {
            match mgr.try_allocate_and_insert(block) {
                Some(block_id) => {
                    self.metrics.stores.fetch_add(1, Ordering::Relaxed);
                    Ok(BlockHandle {
                        level: HierarchyLevel::Chip,
                        addr: block_id as u64,
                        hash: block.meta.hash,
                    })
                }
                None => Err(BlockError::CapacityExhausted),
            }
        }).collect()
    }

    fn remove(&self, hashes: &[BlockHash]) {
        let mut mgr = self.radix_manager.write();
        for h in hashes {
            mgr.remove_by_hash(h);
        }
    }

    fn capacity_blocks(&self) -> u32 { self.radix_manager.read().total_blocks() }
    fn used_blocks(&self) -> u32 { self.radix_manager.read().used_blocks() }
    fn level_name(&self) -> &str { "gpu_hbm" }
}
```

### 6.3 L3 Implementation: `HostBlockStore`

Wraps v7's `LocalBlockIndex` and `SymmetricHeap`:

```rust
/// L3 (Host) block store. Host DRAM and CXL-attached expander.
///
/// Uses v7's LocalBlockIndex for the index and SymmetricHeap for
/// checksummed block I/O. All reads are integrity-verified.
pub struct HostBlockStore {
    local_index: Arc<LocalBlockIndex>,
    heap: Arc<SymmetricHeap>,
    allocator: Arc<dyn LevelAllocator>,
    metrics: LevelMetrics,
}

impl BlockStore for HostBlockStore {
    fn level(&self) -> HierarchyLevel { HierarchyLevel::Host }

    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool> {
        self.metrics.lookups.fetch_add(hashes.len() as u64, Ordering::Relaxed);
        hashes.iter().map(|h| {
            let found = self.local_index.lookup(h).is_some();
            if found {
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.metrics.misses.fetch_add(1, Ordering::Relaxed);
            }
            found
        }).collect()
    }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        hashes.iter().map(|h| {
            let addr = self.local_index.lookup(h)?;

            // Read with checksum verification (v7 §4.4).
            match self.heap.read_block(addr, h) {
                Ok(data) => {
                    self.local_index.ref_inc(h);
                    self.metrics.integrity_checks.fetch_add(1, Ordering::Relaxed);
                    Some(KvBlockData {
                        meta: KvBlockMeta { hash: *h, ..Default::default() },
                        data,
                    })
                }
                Err(e) => {
                    // Corruption detected → quarantine (v7 §7).
                    log::warn!("L3 integrity failure for {:x?}: {:?}", &h[..8], e);
                    self.metrics.integrity_failures.fetch_add(1, Ordering::Relaxed);
                    self.local_index.map.write().remove(h);
                    self.allocator.quarantine(BlockHandle {
                        level: HierarchyLevel::Host,
                        addr: addr.encode(),
                        hash: *h,
                    });
                    None
                }
            }
        }).collect()
    }

    fn store(&self, blocks: &[KvBlockData]) -> Vec<Result<BlockHandle, BlockError>> {
        let allocs = match self.allocator.allocate(blocks.len() as u32) {
            Ok(a) => a,
            Err(_) => return blocks.iter().map(|_| Err(BlockError::CapacityExhausted)).collect(),
        };

        blocks.iter().zip(allocs.iter()).map(|(block, handle)| {
            let addr = GlobalBlockAddr::decode(handle.addr);
            // Write with checksum (v7 §4.3).
            self.heap.write_block(addr, block.meta.hash, &block.data, 0);
            self.local_index.insert_local(block.meta.hash, addr, 0);
            self.metrics.stores.fetch_add(1, Ordering::Relaxed);
            Ok(handle.clone())
        }).collect()
    }

    fn remove(&self, hashes: &[BlockHash]) {
        let mut to_free = Vec::new();
        for h in hashes {
            if let Some(addr) = self.local_index.lookup(h) {
                to_free.push(BlockHandle {
                    level: HierarchyLevel::Host,
                    addr: addr.encode(),
                    hash: *h,
                });
                self.local_index.map.write().remove(h);
            }
        }
        if !to_free.is_empty() {
            self.allocator.free(&to_free);
        }
    }

    fn capacity_blocks(&self) -> u32 { self.allocator.total_count() }
    fn used_blocks(&self) -> u32 { self.allocator.total_count() - self.allocator.free_count() }
    fn level_name(&self) -> &str { "host_dram_cxl" }
}
```

### 6.4 L4–L5 Implementation: `CxlBlockStore`

L4 and L5 use the same implementation as L3 (`HostBlockStore`), differentiated by:
- The CXL region they access (same-switch for L4, cross-switch for L5)
- The gossip scope (L4 gossips within the switch domain, L5 across switch domains)

```rust
/// L4/L5 CXL block store. Same logic as L3 but for pooled/fabric CXL regions.
/// The level is parameterized — same struct serves both L4 and L5.
pub struct CxlBlockStore {
    level: HierarchyLevel,  // SwitchDomain or Fabric
    local_index: Arc<LocalBlockIndex>,
    heap: Arc<SymmetricHeap>,
    allocator: Arc<dyn LevelAllocator>,
    metrics: LevelMetrics,
}

impl BlockStore for CxlBlockStore {
    fn level(&self) -> HierarchyLevel { self.level }
    // ... same implementation as HostBlockStore, parameterized by self.level
    // (omitted for brevity — identical logic, different level tag)
    fn level_name(&self) -> &str {
        match self.level {
            HierarchyLevel::SwitchDomain => "cxl_switch",
            HierarchyLevel::Fabric => "cxl_fabric",
            _ => unreachable!(),
        }
    }
}
```

### 6.5 L6 Implementation: `RemoteBlockStore`

Wraps v5's `RemoteStoreConnector`:

```rust
/// L6 (Global) block store. RDMA/TCP remote store.
pub struct RemoteBlockStore {
    connector: Arc<RemoteStoreConnector>,
    metrics: LevelMetrics,
}

impl BlockStore for RemoteBlockStore {
    fn level(&self) -> HierarchyLevel { HierarchyLevel::Global }

    fn contains(&self, hashes: &[BlockHash]) -> Vec<bool> {
        // Delegates to v5 RemoteStoreConnector::contains
        self.connector.contains(hashes)
    }

    fn fetch(&self, hashes: &[BlockHash]) -> Vec<Option<KvBlockData>> {
        self.connector.fetch(hashes)
    }

    fn store(&self, blocks: &[KvBlockData]) -> Vec<Result<BlockHandle, BlockError>> {
        self.connector.store(blocks);
        blocks.iter().map(|b| Ok(BlockHandle {
            level: HierarchyLevel::Global,
            addr: 0, // remote store manages its own addressing
            hash: b.meta.hash,
        })).collect()
    }

    fn remove(&self, hashes: &[BlockHash]) { self.connector.remove(hashes); }
    fn capacity_blocks(&self) -> u32 { self.connector.capacity() as u32 }
    fn used_blocks(&self) -> u32 { self.connector.usage() as u32 }
    fn level_name(&self) -> &str { "remote_store" }
}
```

### 6.6 Eviction Cascades: L2 → L3 → L4 → L5 → L6 → Discard

When a level is full and needs space, it evicts blocks to the next level up:

```rust
/// Cascading eviction through the hierarchy.
///
/// This is the LLM-serving analog of Linqu's scope-exit retirement:
/// each level retires independently, pushing results to the next level.
pub fn cascade_evict(
    node: &dyn LevelNode,
    parent_store: Option<&dyn BlockStore>,
    count: u32,
) -> u32 {
    // Step 1: Evict from this level's store.
    let evicted = node.evict(count);
    let evicted_count = evicted.len() as u32;

    // Step 2: Push evicted blocks to parent level (if parent exists).
    if let Some(parent) = parent_store {
        let blocks: Vec<KvBlockData> = evicted.into_iter()
            .map(|(hash, data)| KvBlockData {
                meta: KvBlockMeta { hash, ..Default::default() },
                data,
            })
            .collect();
        let results = parent.store(&blocks);
        // Log any failures (parent full → blocks are discarded).
        for (i, r) in results.iter().enumerate() {
            if let Err(e) = r {
                log::debug!("Eviction cascade to {} failed for block: {:?}",
                           parent.level_name(), e);
            }
        }
    }

    evicted_count
}
```

---

## 7. Hierarchical Gossip and Discovery

### 7.1 Same Protocol, Different Scope

v7's `IndexGossip` broadcasts to all peers. v8 scopes gossip by hierarchy level:

```
L3 hosts gossip within their L4 switch domain  (scope: same switch)
L4 leaders gossip within their L5 fabric       (scope: same fabric)
L5 leaders gossip within L6                    (scope: global)
```

The gossip protocol is the SAME at every level. Only the peer list and scope differ.

### 7.2 `LevelGossip` — Level-Parameterized Gossip

```rust
/// Level-parameterized gossip. Same protocol at every level.
/// Replaces v7's flat IndexGossip.
///
/// At each level, the gossip protocol:
/// 1. Periodically broadcasts this node's LevelSummary to peers at the same level.
/// 2. Receives peers' summaries and merges them into the local view.
/// 3. Aggregates children's summaries for the parent level.
pub struct LevelGossip {
    level: HierarchyLevel,
    local_summary: Arc<RwLock<LevelSummary>>,
    /// Summaries from peer nodes at the same level.
    peer_summaries: Arc<RwLock<HashMap<u32, LevelSummary>>>,
    /// UDP socket for gossip.
    socket: UdpSocket,
    /// Peers at the same level.
    peers: Vec<SocketAddr>,
    /// Gossip interval (level-specific: shorter for lower levels).
    interval: Duration,
}

impl LevelGossip {
    /// Broadcast this node's summary to peers.
    pub fn broadcast(&self) {
        let summary = self.local_summary.read().clone();
        let bytes = bincode::serialize(&summary).unwrap();
        for peer in &self.peers {
            let _ = self.socket.send_to(&bytes, peer);
        }
    }

    /// Receive peer summaries.
    pub fn receive(&self) {
        let mut buf = vec![0u8; 65536];
        while let Ok((n, _)) = self.socket.recv_from(&mut buf) {
            if let Ok(summary) = bincode::deserialize::<LevelSummary>(&buf[..n]) {
                self.peer_summaries.write().insert(summary.node_id, summary);
            }
        }
    }

    /// Aggregate: combine this node's summary with children's summaries
    /// to produce a summary for the parent level.
    pub fn aggregate(&self, children: &[Arc<dyn LevelNode>]) -> LevelSummary {
        let mut combined_hashes = Vec::new();
        let mut combined_count = 0u32;
        let mut combined_free = 0u32;

        // This node's own blocks.
        let local = self.local_summary.read();
        combined_count += local.block_count;
        combined_free += local.free_blocks;

        // Children's blocks.
        for child in children {
            let child_summary = child.summary();
            combined_count += child_summary.block_count;
            combined_free += child_summary.free_blocks;
            // For large block sets, use a bloom filter instead of full hash list.
            combined_hashes.extend_from_slice(&child_summary.block_hashes);
        }

        LevelSummary {
            level: self.level.parent().unwrap_or(self.level),
            node_id: local.node_id,
            block_hashes: combined_hashes,
            block_count: combined_count,
            free_blocks: combined_free,
            timestamp_ms: unix_timestamp_ms(),
        }
    }
}
```

### 7.3 Gossip Intervals by Level

| Level | Default Interval | Rationale |
|---|---|---|
| L3 (Host → L4 leader) | 1 ms | Low latency within switch domain; blocks change frequently |
| L4 (L4 leader → L5 leader) | 5 ms | Aggregated summary, less frequent changes |
| L5 (L5 leader → L6) | 10 ms | Cross-fabric, even less frequent |

### 7.4 Collapse Behavior

For small clusters where L4 = L5 = L6 is collapsed, all hosts gossip directly to each other at the L3 interval. This matches v7's flat gossip behavior. The hierarchical structure exists in the code but produces flat behavior when the hierarchy is flat.

---

## 8. Uniform Integrity Model

### 8.1 Same Interface, Level-Specific Implementation

Every level has an `IntegrityVerifier` (§3.6). The interface is always:

```
verify(block) → Ok(()) | Err(IntegrityError)
scrub() → count_of_corrupted_blocks
quarantine(block)
```

### 8.2 Per-Level Defaults

| Level | Checksum | Verify on Read | Scrub | Quarantine | Rationale |
|---|---|---|---|---|---|
| L2 (GPU) | None (ECC) | No | Optional | Mark GPU block unusable | GPU HBM is ECC-protected. Software checksums would add latency on the hottest path. |
| L3 (Host DRAM) | None (ECC) | No | No | N/A | Host DRAM is ECC-protected like GPU HBM. |
| L3–L5 (CXL) | xxHash64 | Yes | Yes (v7 §8) | Yes (v7 §7.3) | CXL memory traverses more hops than local DRAM. v7's control/data split applies here. |
| L6 (Remote) | None (TCP) | No | No | Remove remote key | TCP checksum handles transport integrity. |

### 8.3 v7's Control/Data Plane Split in the Recursive Model

v7's key architectural insight — separate control plane (local memory) from data plane (CXL shared memory) — is preserved as an implementation detail of the L3–L5 `BlockStore` and `IntegrityVerifier`:

- **Control plane** (local): `LocalBlockIndex`, `CxlMemoryManager`, refcounts → all local DRAM, protected by local ECC.
- **Data plane** (CXL): KV block payload → CXL shared memory, protected by xxHash64 checksum.

This split is invisible to the `LevelNode` trait. It is an implementation choice for levels where the control/data distinction matters. At L2 and L6, the distinction doesn't apply (everything is either local or remote), so the implementation is simpler.

---

## 9. Recursive Failure Handling

### 9.1 The Pattern: Detect → Quarantine → Recover

Every level follows the same failure pattern:

```
1. DETECT:     Level-specific mechanism detects a problem.
2. QUARANTINE: Remove the affected resource from use.
3. RECOVER:    Either recompute (KV is deterministic) or failover.
4. ESCALATE:   If local recovery fails, notify parent level.
```

### 9.2 Per-Level Implementation

| Level | Detect | Quarantine | Recover | Escalate |
|---|---|---|---|---|
| L2 (GPU) | ECC error interrupt, CUDA error on kernel launch | Mark GPU block as unusable in `BlockPool` | Reallocate a new GPU block, recompute KV | If GPU is dead: remove `ChipNode` from parent `HostNode`'s children |
| L3 (Host) | xxHash64 checksum failure on CXL read (v7 §7) | `CxlMemoryManager::quarantine()` — block stays allocated but unusable (v7 §7.3) | Recompute the block (KV cache miss) | If too many quarantined blocks: alert and consider draining the CXL device |
| L4 (Switch) | Gossip timeout — a host in the switch domain stops responding | Remove host from peer list; redistribute its blocks' index entries | Blocks on the failed host are lost (cache miss → recompute) | If entire switch domain is unhealthy: parent L5 node reroutes |
| L5 (Fabric) | Gossip timeout from L4 leader | Remove switch domain from active set | Blocks in that domain are lost (recompute) | Parent L6 node reroutes to different fabric |
| L6 (Global) | RPC timeout to remote store | Mark remote store as unavailable | Fall back to local-only caching (no L6 tier) | N/A — top of hierarchy |

### 9.3 Failure Escalation

When a child node fails, the parent adjusts:

```rust
impl LevelNode for ClusterNode {
    fn health(&self) -> LevelHealth {
        let children = self.children();
        let healthy = children.iter()
            .filter(|c| c.health().status == HealthStatus::Healthy)
            .count() as u32;
        let total = children.len() as u32;

        LevelHealth {
            level: self.level(),
            status: if healthy == total { HealthStatus::Healthy }
                    else if healthy > 0 { HealthStatus::Degraded }
                    else { HealthStatus::Failed },
            healthy_children: healthy,
            unhealthy_children: total - healthy,
            integrity_failures: self.metrics.integrity_failures.load(Ordering::Relaxed),
            quarantined_blocks: self.verifier.quarantined_count(),
        }
    }

    fn route(&self, hashes: &[BlockHash], load: &LoadSnapshot) -> usize {
        let children = self.children();
        // Skip unhealthy children.
        let candidates: Vec<(usize, &Arc<dyn LevelNode>)> = children.iter()
            .enumerate()
            .filter(|(_, c)| c.health().status != HealthStatus::Failed)
            .collect();

        if candidates.is_empty() {
            // All children failed — escalate to parent by returning 0
            // and letting the parent's health check detect the failure.
            return 0;
        }

        // Score only healthy candidates.
        candidates.iter()
            .max_by(|(i, a), (j, b)| {
                let score_a = score_child(a.as_ref(), hashes, load, *i, &self.routing_params);
                let score_b = score_child(b.as_ref(), hashes, load, *j, &self.routing_params);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| *idx)
            .unwrap_or(0)
    }
}
```

---

## 10. Hierarchical Capacity Management and Eviction

### 10.1 Per-Level Watermarks

Each level has high/low watermarks for eviction (from `EvictionConfig`):

```
Level:     L2 (GPU)    L3 (Host)    L4 (Switch)  L5 (Fabric)  L6 (Global)
High WM:   90%         90%          95%          95%          99%
Low WM:    70%         70%          80%          80%          90%
```

When a level exceeds its high watermark, eviction starts and pushes blocks to the parent level. Eviction stops when usage drops below the low watermark.

### 10.2 The Eviction Cascade

```
GPU 95% full → evict GPU blocks → push to Host DRAM/CXL
  Host 92% full → evict Host blocks → push to CXL pool (L4)
    CXL pool 96% full → evict CXL blocks → push to CXL fabric (L5)
      CXL fabric 96% full → evict fabric blocks → push to remote store (L6)
        Remote store 99% full → evict remote blocks → DISCARD
```

Each level decides WHICH blocks to evict using its configured `EvictionPolicy`:
- L2: `LongestUnusedPrefix` (radix tree walk, v4 design)
- L3–L5: `Lru` (least recently used)
- L6: `Lru` or `Discard`

### 10.3 Promotion and Demotion

- **Promotion** (higher level → lower level): When a request needs a block, the router descends the tree and finds it at, say, L4. The block is promoted down to L2 (GPU) via L3 (host DRAM) as an intermediate step if needed.
- **Demotion** (lower level → higher level): When a level is full, eviction pushes blocks up.

This replaces v7's `promote_from_cxl` and `evict_gpu_to_cxl` with a generic mechanism:

```rust
/// Promote a block from a higher level to a lower level.
/// Follows the hierarchy: L5 → L4 → L3 → L2.
pub fn promote_block(
    root: &dyn LevelNode,
    hash: &BlockHash,
    target_level: HierarchyLevel,
) -> Result<BlockHandle, BlockError> {
    // Find the block in the hierarchy (search from top).
    let (source_level, data) = find_block_in_hierarchy(root, hash)?;

    // The block needs to descend from source_level to target_level.
    // Each intermediate level stores a copy (for future hits at that level).
    let mut current_data = data;
    let mut current_level = source_level;
    let mut current_handle = None;

    while current_level > target_level {
        let child_level = current_level.child().unwrap();
        let child_node = find_node_at_level(root, child_level);
        let result = child_node.store().store(&[KvBlockData {
            meta: KvBlockMeta { hash: *hash, ..Default::default() },
            data: current_data.clone(),
        }]);
        current_handle = result.into_iter().next().and_then(|r| r.ok());
        current_level = child_level;
    }

    current_handle.ok_or(BlockError::CapacityExhausted)
}
```

---

## 11. What Changes from v7

### 11.1 Component-by-Component Table

| v7 Component | v8 Change | Status |
|---|---|---|
| `RadixKvManager` + `BlockPool` (v4 §9) | Wrapped behind `GpuBlockStore` implementing `BlockStore` trait | **Wrapped** (internals unchanged) |
| `LocalBlockIndex` (v7 §5.1) | Wrapped behind `HostBlockStore` implementing `BlockStore` trait | **Wrapped** (internals unchanged) |
| `CxlKvConnector` (v7 §11) | **Replaced** by `CxlBlockStore` implementing `BlockStore` trait | **Replaced** |
| `RemoteStoreConnector` (v5 §4.4) | Wrapped behind `RemoteBlockStore` implementing `BlockStore` trait | **Wrapped** (internals unchanged) |
| `CxlAwareRouter` (v7 §13.1) | **Replaced** by recursive `LevelNode::route()` at each level | **Replaced** |
| `IndexGossip` (v7 §5.2) | **Replaced** by hierarchical `LevelGossip` parameterized by level | **Replaced** |
| `CxlAwarePoolManager` (v7 §10) | **Replaced** by recursive hierarchy tree. The tree IS the pool manager. | **Replaced** |
| `CxlMemoryManager` (v7 §6.2) | Wrapped behind `LevelAllocator` trait for L3–L5 | **Wrapped** (internals unchanged) |
| `AllocatorStandby` (v7 §6.4) | Unchanged — implementation detail of L4 `LevelAllocator` | **Unchanged** |
| `CxlScrubber` (v7 §8) | Wrapped behind `IntegrityVerifier` trait for L3–L5 | **Wrapped** (internals unchanged) |
| `UdpHeartbeat` (v7 §9) | Unchanged — implementation detail of L3–L5 health check | **Unchanged** |
| `CxlBlockHeader` + checksummed I/O (v7 §4) | Unchanged — implementation detail of L3–L5 `BlockStore` | **Unchanged** |
| `SharedWeightPool` (v7 §12) | Unchanged — orthogonal to KV block hierarchy | **Unchanged** |
| `CxlTopology::discover()` (v6 §2) | Unchanged — input to `build_hierarchy_tree()` | **Unchanged** |
| Configuration schema | **Restructured** from flat level-specific sections to recursive `levels[L]` | **Restructured** |
| Metrics | **Restructured** from ad-hoc to uniform `LevelMetrics[L]` | **Restructured** |
| All v4 components (HAL, scheduler, forward pass, etc.) | Unchanged | **Unchanged** |

### 11.2 New Abstractions

| New Type | Purpose |
|---|---|
| `HierarchyLevel` enum | Linqu-aligned level identifier |
| `BlockStore` trait | Unified storage interface (replaces `KvConnector`) |
| `LevelNode` trait | Recursive hierarchy node |
| `LevelAllocator` trait | Uniform block allocation |
| `IntegrityVerifier` trait | Uniform integrity checking |
| `LevelConfig` struct | Per-level configuration |
| `LevelMetrics` struct | Per-level metrics |
| `LevelGossip` struct | Hierarchical gossip |
| `LevelSummary` struct | Gossip summary for parent aggregation |
| `BlockHandle` struct | Level-aware block reference |
| `build_hierarchy_tree()` | Tree construction from topology |
| `route_request()` | Recursive routing through the tree |
| `cascade_evict()` | Cascading eviction up the hierarchy |
| `promote_block()` | Promotion down the hierarchy |

---

## 12. Risks and Mitigations

### 12.1 GPU Radix Tree vs. Flat Hash Maps

**Risk:** The GPU's `RadixKvManager` uses a radix tree for prefix sharing and tree-walk eviction. Higher levels use flat hash maps. The `BlockStore` trait must accommodate both.

**Mitigation:** The `BlockStore` trait provides the **common interface** (contains, fetch, store, remove). The radix tree is an internal optimization of `GpuBlockStore`, invisible to the trait. This is the same pattern as Linqu: Level 0 has TPUSH/TPOP that higher levels don't — level-specific capabilities exist inside the uniform interface, not outside it.

### 12.2 Centralized CXL Allocator

**Risk:** v7's `CxlMemoryManager` is centralized (one primary, one standby). In a recursive model, each L4 `SwitchDomainNode` would ideally have its own allocator. But per-level allocation in CXL shared memory was rejected in v7 due to corruption risk.

**Mitigation:** Keep the centralized `CxlMemoryManager` as the `LevelAllocator` implementation for L3–L5. Logically, it belongs at L4 (the switch domain level, since CXL memory regions are scoped to switch domains). L3 nodes request allocations from their L4 parent's allocator via RPC. This fits the hierarchy: allocation authority is at L4, access is at L3–L5.

### 12.3 Routing Overhead for Simple Topologies

**Risk:** For a 2-host cluster (4 GPUs total), recursive routing evaluates 3 levels of `route()` calls when flat routing would suffice.

**Mitigation:** Level collapsing (§4.2) reduces the tree to `GlobalNode → HostNode → ChipNode` (3 hops). Each hop's `route()` is a simple argmax over 2 children — negligible overhead. The recursive structure adds ~10ns of function call overhead compared to flat routing. On the critical path (request routing), this is unmeasurable against ~10ms prefill latency.

### 12.4 Forced Uniformity vs. Level-Specific Concerns

**Risk:** v7's control/data plane split is critical for CXL reliability but not relevant at L2 or L6. Forcing the same abstraction everywhere could obscure this.

**Mitigation:** The control/data split is an **implementation detail** of the L3–L5 `BlockStore`, not a trait-level concern. The `IntegrityConfig` per level captures the relevant differences (checksum algo, verify-on-read, scrub). The v8 document explicitly notes: "the recursive structure parameterizes level-specific concerns, it does not flatten them away."

### 12.5 Gossip Complexity at Small Scale

**Risk:** Hierarchical gossip requires leader election. For 2-host clusters, this is overhead without benefit.

**Mitigation:** At small scale, every host is its own L4/L5/L6 leader (level collapsed). The `LevelGossip` code exists at every level, but with collapsed levels it degenerates to flat gossip. No leader election is needed because there is only one node per level.

---

## 13. Configuration

### 13.1 Recursive `levels[L]` Schema

v8 replaces v7's flat, level-specific configuration with a uniform per-level schema:

```yaml
# Each level has the same configuration schema with different defaults.
hierarchy:
  levels:
    - level: chip          # L2
      enabled: true
      # Capacity is auto-detected from GPU memory profiling (v4 §7).
      integrity:
        checksum: none     # ECC handles it
        verify_on_read: false
        scrub_enabled: false
      eviction:
        high_watermark: 0.90
        low_watermark: 0.70
        policy: longest_unused_prefix
      routing:
        hit_weight: 10.0
        load_weight: 1.0
        capacity_weight: 0.5

    - level: host          # L3
      enabled: true
      capacity_blocks: 100000   # ~3.2 GB at 32KB/block
      block_size_bytes: 32768
      integrity:
        checksum: xxhash64
        verify_on_read: true
        scrub_enabled: true
        scrub_interval_secs: 120
      gossip:
        listen_addr: "0.0.0.0:9300"
        interval_ms: 1
        peers: []          # auto-discovered from CXL topology
      health_check:
        interval_ms: 100
        timeout_ms: 500
      eviction:
        high_watermark: 0.90
        low_watermark: 0.70
        policy: lru
      routing:
        hit_weight: 5.0
        load_weight: 2.0
        capacity_weight: 1.0

    - level: switch_domain   # L4
      enabled: true
      capacity_blocks: 0     # auto from CXL topology
      integrity:
        checksum: xxhash64
        verify_on_read: true
        scrub_enabled: true
        scrub_interval_secs: 120
      gossip:
        listen_addr: "0.0.0.0:9301"
        interval_ms: 5
        peers: []
      eviction:
        high_watermark: 0.95
        low_watermark: 0.80
        policy: lru
      routing:
        hit_weight: 3.0
        load_weight: 3.0
        capacity_weight: 1.0

    - level: fabric          # L5
      enabled: false         # enable for multi-switch deployments
      # ... same schema as L4 with different defaults

    - level: global          # L6
      enabled: false         # enable for multi-fabric deployments
      integrity:
        checksum: none       # TCP handles it
        verify_on_read: false
      eviction:
        high_watermark: 0.99
        low_watermark: 0.90
        policy: lru

  # Centralized allocator (v7, unchanged).
  allocator:
    is_primary: false
    primary_addr: "host0:9200"
    is_standby: false
    standby_addr: "host1:9200"

  # Heartbeat (v7, unchanged).
  heartbeat:
    listen_addr: "0.0.0.0:9100"
    interval_ms: 100
    timeout_ms: 500

# CXL topology discovery (v6, unchanged).
cxl:
  enabled: true
  topology:
    auto_discover: true

# Shared weight pool (v7, unchanged).
weight_pool:
  enabled: true
  is_leader: false
  verify_interval_secs: 300

# Legacy v5 settings for non-CXL fallback.
kv_cache_pool:
  enabled: true
  cpu_dram:
    enabled: true
    capacity_gb: 32
```

### 13.2 Minimal Config (2 Hosts, Single CXL Switch)

```yaml
# Host 0 (allocator primary + weight leader):
hierarchy:
  levels:
    - { level: chip, enabled: true }
    - { level: host, enabled: true, capacity_blocks: 100000,
        gossip: { peers: ["host1:9300"] } }
    - { level: switch_domain, enabled: true,
        gossip: { peers: ["host1:9301"] } }
  allocator: { is_primary: true, primary_addr: "host0:9200" }
cxl: { enabled: true }
weight_pool: { enabled: true, is_leader: true }

# Host 1 (allocator standby):
hierarchy:
  levels:
    - { level: chip, enabled: true }
    - { level: host, enabled: true, capacity_blocks: 100000,
        gossip: { peers: ["host0:9300"] } }
    - { level: switch_domain, enabled: true,
        gossip: { peers: ["host0:9301"] } }
  allocator: { is_standby: true, primary_addr: "host0:9200",
               standby_addr: "host1:9200" }
cxl: { enabled: true }
weight_pool: { enabled: true, is_leader: false }
```

---

## 14. Updated Directory Layout

```
llm-server/src/
├── hierarchy/                  ← NEW: recursive hierarchy framework
│   ├── mod.rs                  ← HierarchyLevel, BlockHandle, LevelConfig
│   ├── traits.rs               ← BlockStore, LevelNode, LevelAllocator, IntegrityVerifier
│   ├── metrics.rs              ← LevelMetrics (uniform per-level)
│   ├── tree.rs                 ← build_hierarchy_tree(), level collapsing
│   ├── routing.rs              ← recursive routing (route_request, score_child)
│   ├── eviction.rs             ← cascade_evict(), promote_block()
│   └── gossip.rs               ← LevelGossip (hierarchical, level-parameterized)
├── hierarchy/levels/           ← NEW: per-level implementations
│   ├── chip.rs                 ← ChipNode, GpuBlockStore, GpuAllocator
│   ├── host.rs                 ← HostNode, HostBlockStore
│   ├── cluster.rs              ← ClusterNode, CxlBlockStore (L4/L5/L6)
│   └── remote.rs               ← RemoteBlockStore (L6)
├── cxl/                        ← v7 CXL internals (UNCHANGED)
│   ├── topology.rs             ← CxlTopology (v6, unchanged)
│   ├── symmetric_heap.rs       ← SymmetricHeap (v7, unchanged)
│   ├── block_header.rs         ← CxlBlockHeader, checksummed I/O (v7, unchanged)
│   ├── local_index.rs          ← LocalBlockIndex (v7, unchanged)
│   ├── allocator.rs            ← CxlMemoryManager (v7, unchanged)
│   ├── allocator_client.rs     ← AllocatorClient (v7, unchanged)
│   ├── allocator_standby.rs    ← AllocatorStandby (v7, unchanged)
│   ├── scrubber.rs             ← CxlScrubber (v7, unchanged)
│   ├── heartbeat.rs            ← UdpHeartbeat (v7, unchanged)
│   ├── weight_pool.rs          ← SharedWeightPool (v7, unchanged)
│   └── quarantine.rs           ← Quarantine tracking (v7, unchanged)
├── kv_cache/                   ← v4 internals (UNCHANGED)
│   ├── radix_tree.rs           ← RadixKvManager (v4, unchanged)
│   ├── block_pool.rs           ← BlockPool (v4, unchanged)
│   └── ...
├── routing/
│   ├── router.rs               ← REPLACED by hierarchy/routing.rs
│   └── ...                     ← unchanged
├── scheduler/                  ← v4, unchanged
├── executor/                   ← v4, unchanged
├── hal/                        ← v4, unchanged
└── ...                         ← v4/v5, unchanged
```

### New/Changed Crate Dependencies

| Crate | Purpose | Status |
|---|---|---|
| `xxhash-rust` | Block checksums (v7, unchanged) | Unchanged |
| `bincode` | Gossip + allocator RPC (v7, unchanged) | Unchanged |
| `memmap2` | CXL device mmap (v6, unchanged) | Unchanged |
| No new dependencies | v8 is a structural refactoring, not a new capability | — |

---

## 15. Implementation Phases

### Phase 12a — Core Traits and Tree Construction (1 week)

#### Deliverables

- [ ] `HierarchyLevel` enum with `child()`/`parent()` methods
- [ ] `BlockStore` trait definition
- [ ] `LevelNode` trait definition
- [ ] `LevelAllocator` trait definition
- [ ] `IntegrityVerifier` trait definition
- [ ] `LevelConfig`, `LevelMetrics`, `LevelSummary` structs
- [ ] `BlockHandle`, `BlockError`, `IntegrityError` types
- [ ] `build_hierarchy_tree()` from CXL topology + config
- [ ] Level collapsing logic
- [ ] Unit tests: tree construction, level collapsing, hierarchy traversal

#### Test Plan

**T12a.1 — Hierarchy level traversal:**
```rust
#[test]
fn test_hierarchy_level_parent_child() {
    assert_eq!(HierarchyLevel::Chip.parent(), Some(HierarchyLevel::Host));
    assert_eq!(HierarchyLevel::Host.child(), Some(HierarchyLevel::Chip));
    assert_eq!(HierarchyLevel::Global.parent(), None);
    assert_eq!(HierarchyLevel::Chip.child(), None);
}
```

**T12a.2 — Level collapsing (single switch, 2 hosts):**
```rust
#[test]
fn test_level_collapsing_2_hosts() {
    let config = vec![
        LevelConfig { level: HierarchyLevel::Chip, enabled: true, .. },
        LevelConfig { level: HierarchyLevel::Host, enabled: true, .. },
        LevelConfig { level: HierarchyLevel::SwitchDomain, enabled: true, .. },
        LevelConfig { level: HierarchyLevel::Fabric, enabled: false, .. },
        LevelConfig { level: HierarchyLevel::Global, enabled: false, .. },
    ];
    let tree = build_hierarchy_tree(&mock_2host_topology(), &config, 0);

    // Root should be SwitchDomain (L4), with L5/L6 collapsed.
    assert_eq!(tree.level(), HierarchyLevel::SwitchDomain);
    assert_eq!(tree.children().len(), 2);  // 2 hosts
    assert_eq!(tree.children()[0].level(), HierarchyLevel::Host);
}
```

**T12a.3 — Full tree construction (4 hosts, 2 switches):**
```rust
#[test]
fn test_full_tree_4_hosts_2_switches() {
    let config = all_levels_enabled();
    let tree = build_hierarchy_tree(&mock_4host_2switch_topology(), &config, 0);

    // Root: Global (L6) → Fabric (L5) → 2 SwitchDomain (L4) → 2 Host (L3) each → N Chip (L2)
    assert_eq!(tree.level(), HierarchyLevel::Global);
    let fabric = &tree.children()[0];
    assert_eq!(fabric.level(), HierarchyLevel::Fabric);
    let switches = fabric.children();
    assert_eq!(switches.len(), 2);
    for sw in switches {
        assert_eq!(sw.level(), HierarchyLevel::SwitchDomain);
        assert_eq!(sw.children().len(), 2);  // 2 hosts per switch
    }
}
```

---

### Phase 12b — Wrap Existing Implementations Behind Traits (1.5 weeks)

#### Deliverables

- [ ] `GpuBlockStore` implementing `BlockStore` (wraps `RadixKvManager`)
- [ ] `HostBlockStore` implementing `BlockStore` (wraps `LocalBlockIndex` + `SymmetricHeap`)
- [ ] `CxlBlockStore` implementing `BlockStore` (parameterized for L4/L5)
- [ ] `RemoteBlockStore` implementing `BlockStore` (wraps `RemoteStoreConnector`)
- [ ] `GpuAllocator` implementing `LevelAllocator`
- [ ] `CxlAllocatorClient` implementing `LevelAllocator`
- [ ] `GpuIntegrityVerifier`, `CxlIntegrityVerifier`, `RemoteIntegrityVerifier`
- [ ] `ChipNode`, `HostNode`, `ClusterNode` implementing `LevelNode`
- [ ] All existing v7 tests pass through the new trait interface

#### Test Plan

**T12b.1 — BlockStore trait roundtrip at every level:**
```rust
#[test]
fn test_blockstore_roundtrip_all_levels() {
    for store in [gpu_store(), host_store(), cxl_store(), remote_store()] {
        let hash = test_hash(42);
        let data = vec![0xABu8; 32768];
        let block = KvBlockData { meta: KvBlockMeta { hash, .. }, data: data.clone() };

        // Store.
        let results = store.store(&[block]);
        assert!(results[0].is_ok());

        // Contains.
        assert_eq!(store.contains(&[hash]), vec![true]);

        // Fetch.
        let fetched = store.fetch(&[hash]);
        assert_eq!(fetched[0].as_ref().unwrap().data, data);

        // Remove.
        store.remove(&[hash]);
        assert_eq!(store.contains(&[hash]), vec![false]);
    }
}
```

**T12b.2 — LevelMetrics uniformity:**
```rust
#[test]
fn test_level_metrics_uniform_across_levels() {
    for node in [chip_node(), host_node(), cluster_node()] {
        let m = node.metrics();
        // Same metric fields available at every level.
        assert_eq!(m.lookups.load(Ordering::Relaxed), 0);
        assert_eq!(m.hits.load(Ordering::Relaxed), 0);
        assert_eq!(m.integrity_failures.load(Ordering::Relaxed), 0);
    }
}
```

---

### Phase 12c — Recursive Routing (1 week)

#### Deliverables

- [ ] `score_child()` function with per-level `RoutingParams`
- [ ] `route_request()` recursive descent
- [ ] `LevelNode::route()` implementation for all node types
- [ ] Routing benchmarks comparing v7 flat vs v8 recursive

#### Test Plan

**T12c.1 — Recursive routing prefers cache-hit child:**
```rust
#[test]
fn test_recursive_routing_prefers_cache_hit() {
    let tree = build_test_tree_2_hosts_4_gpus();

    // Store a block on GPU 2 (host 1, chip 0).
    tree.children()[1].children()[0].store()
        .store(&[test_block(42)]);

    let path = route_request(tree.as_ref(), &[test_hash(42)], &balanced_load());

    // Should route to host 1 (index 1), chip 0 (index 0).
    assert_eq!(path, vec![1, 0]);
}
```

**T12c.2 — Routing skips failed children:**
```rust
#[test]
fn test_routing_skips_failed_host() {
    let tree = build_test_tree_2_hosts();
    // Mark host 0 as failed.
    tree.children()[0].set_health(HealthStatus::Failed);

    let path = route_request(tree.as_ref(), &[test_hash(42)], &balanced_load());

    // Should route to host 1 (index 1).
    assert_eq!(path[0], 1);
}
```

---

### Phase 12d — Hierarchical Gossip (1 week)

#### Deliverables

- [ ] `LevelGossip` struct with `broadcast()`, `receive()`, `aggregate()`
- [ ] Per-level gossip intervals
- [ ] Summary aggregation from children
- [ ] Tests: gossip convergence, hierarchical aggregation

#### Test Plan

**T12d.1 — Hierarchical summary aggregation:**
```rust
#[test]
fn test_hierarchical_summary_aggregation() {
    // L3 host 0 has 100 blocks, host 1 has 50 blocks.
    let host0 = test_host_node(100);
    let host1 = test_host_node(50);

    // L4 switch domain aggregates.
    let switch_gossip = LevelGossip::new(HierarchyLevel::SwitchDomain, ..);
    let summary = switch_gossip.aggregate(&[host0, host1]);

    assert_eq!(summary.block_count, 150);
}
```

---

### Phase 12e — Integration Test: Full Recursive Pipeline (1 week)

#### Deliverables

- [ ] End-to-end test: request arrives → recursive route → fetch from L4 → promote to L2
- [ ] End-to-end test: GPU full → cascading eviction L2 → L3 → L4
- [ ] End-to-end test: CXL corruption during promotion → detect → quarantine → recompute
- [ ] Benchmark: v7 flat vs v8 recursive routing throughput
- [ ] Benchmark: v7 flat vs v8 hierarchical gossip bandwidth

#### Test Plan

**T12e.1 — Full promotion pipeline (L4 → L3 → L2):**
```rust
#[test]
fn test_full_promotion_l4_to_l2() {
    let tree = build_test_tree_2_hosts_cxl();

    // Store a block at L4 (CXL pool).
    let l4_store = tree.children()[0].store();  // switch domain
    l4_store.store(&[test_block(42)]);

    // Promote to L2 (GPU).
    let handle = promote_block(tree.as_ref(), &test_hash(42), HierarchyLevel::Chip);
    assert!(handle.is_ok());
    assert_eq!(handle.unwrap().level, HierarchyLevel::Chip);

    // Block should now exist at L2.
    let gpu_store = find_chip_node(tree.as_ref()).store();
    assert_eq!(gpu_store.contains(&[test_hash(42)]), vec![true]);
}
```

**T12e.2 — Cascading eviction (L2 → L3 → L4):**
```rust
#[test]
fn test_cascading_eviction() {
    let tree = build_test_tree_small_capacity();

    // Fill L2 (GPU) beyond high watermark.
    let gpu_store = find_chip_node(tree.as_ref()).store();
    for i in 0..100 {
        gpu_store.store(&[test_block(i)]);
    }

    // Trigger eviction at L2.
    let evicted = cascade_evict(
        find_chip_node(tree.as_ref()),
        Some(find_host_node(tree.as_ref()).store()),
        20,
    );
    assert_eq!(evicted, 20);

    // Evicted blocks should be at L3 now.
    let host_store = find_host_node(tree.as_ref()).store();
    assert!(host_store.used_blocks() >= 20);
}
```

---

## 16. Performance Targets

### v8 vs. v7: Overhead of Recursive Structure

| Metric | v7 | v8 | Delta |
|---|---|---|---|
| Routing decision (per request) | ~100ns (flat score N instances) | ~150ns (recursive, 3–5 levels × argmax) | **+50ns** (+50%, negligible vs 10ms prefill) |
| Gossip bandwidth (8 hosts) | ~1 MB/s per host (flat) | ~0.5 MB/s per host (hierarchical, smaller per-level deltas) | **-50%** (better at scale) |
| Block promotion (CXL → GPU) | ~4μs (v7 checksum + DMA) | ~4μs (same, trait dispatch is ~0ns) | **~0** |
| Memory overhead (trait vtables) | 0 | ~64 bytes per `LevelNode` (vtable + Arc overhead) | **Negligible** |
| Configuration complexity | Level-specific sections | Uniform `levels[L]` schema | **Simpler** (same schema, different defaults) |
| Monitoring/alerting | Per-component metrics | Uniform `LevelMetrics[L]` | **Simpler** (one alert rule for all levels) |

### v8 vs. v5: Net Performance (What Matters to Users)

Same as v7 vs. v5 — the recursive structure does not change data-path performance:

| Metric | v5 (Copy-Based) | v8 (CXL + Recursive) |
|---|---|---|
| Block promotion latency | ~15μs | ~4μs (3.75× faster) |
| Cross-instance KV fetch | ~50μs | ~9μs worst case (5.5× faster) |
| 131K-token PD transfer | ~140–510ms | ~57ms (2.5–9× faster) |
| Silent corruption probability | No detection | ~0 (checksum + scrubber) |

### Scaling Improvement

| Cluster Size | v7 Routing Complexity | v8 Routing Complexity |
|---|---|---|
| 4 instances (1 host, 4 GPUs) | O(4) | O(2+4) = O(6) |
| 16 instances (2 hosts, 8 GPUs each) | O(16) | O(2+8) = O(10) |
| 64 instances (8 hosts, 8 GPUs each) | O(64) | O(2+8+8) = O(18) |
| 512 instances (64 hosts, 8 GPUs each) | O(512) | O(8+8+8+8) = O(32) |

---

## 17. Mapping to Linqu Hierarchy — Structural, Not Tabular

v7's §18 was a table: "this component maps to that level." v8 makes the mapping **structural** — the hierarchy is not just a documentation artifact, it is the code architecture.

### 17.1 How Each Linqu Principle Is Realized

| Linqu Principle | Realization in v8 |
|---|---|
| **Hierarchical symmetry** (§1.1) | Every level implements the same `LevelNode` + `BlockStore` + `LevelAllocator` traits. There is no component that exists at one level but not another. |
| **Recursive enclosure** (§2.2) | `LevelNode::children()` returns the next-level-down nodes. An L4 node encloses L3 nodes. An L3 node encloses L2 nodes. The runtime tree mirrors the physical hierarchy. |
| **Level-parameterized data structures** (§5.1) | `block_pool[L]` — every level has a `BlockStore` with `contains`/`fetch`/`store`/`remove` and the same `LevelMetrics` schema. Just like Linqu's `buffer_ring[L][d]`. |
| **Unified programming model** (§1.6) | `route_request()` descends the tree uniformly. `cascade_evict()` ascends the tree uniformly. Adding a new level means implementing `LevelNode` — the framework handles routing, eviction, gossip, and metrics automatically. |
| **Three-tier communication** (§7.4) | L2–L3: GPU DMA (`cudaMemcpyAsync`). L3–L5: CXL load/store (checksummed). L5–L6: RPC (TCP/RDMA). Different transport, same `BlockStore::fetch`/`store` interface. |
| **Forward-compatible design** (§3.5) | All data structures accept `HierarchyLevel` for any level. Config has entries for L2–L6. Tests exercise the full hierarchy with mock topologies. Adding L7 (or inserting L1 for multi-die GPUs) requires implementing one new `LevelNode` and adding a config entry — no changes to the framework. |

### 17.2 What Linqu Concepts Are NOT Used (and Why)

| Linqu Concept | Why Not Used | What We Use Instead |
|---|---|---|
| `task_ring[L][d]` | LLM serving uses request-level continuous batching, not scope-based task submission. | `BlockStore` with hash-based block identity. |
| `scope_depth d` | KV blocks have flat lifetime (LRU eviction), not nested scope lifetime. | `EvictionPolicy` per level. |
| `pl.free(tensor)` | KV blocks are recomputable caches. Lifetime is managed by LRU, not explicit free. | `BlockStore::remove()` + LRU eviction. |
| SPMD fan-out | LLM requests are routed to specific instances, not broadcast. | Recursive `route_request()` descent. |
| `TaskKey(scope_level, task_id)` | KV blocks are identified by content hash (token prefix SHA-256). | `BlockHash` — the universal block identity. |

---

## 18. Meta Index Durability and Cost at TB-Scale

### 18.1 The Problem: Recomputation Is Not Free at TB Scale

v7 and v8 §9.2 assume KVCache blocks are "recomputable caches" — if a host fails, its blocks are lost and will be recomputed on demand. This assumption holds when KVCache is measured in tens of GB. At **several TB** of KVCache (e.g., 64 hosts × 128 GB CXL per host = 8 TB total), recomputation becomes prohibitively expensive:

| Scenario | Recompute Cost |
|---|---|
| Single host loss (128 GB KV) | ~50–200B tokens × forward pass cost. At 100K tok/s, this is **minutes to hours** of recomputation while serving capacity is degraded. |
| CXL switch failure (1 TB KV, 8 hosts) | Cannot serve requests that depend on those cached prefixes. Cold-start penalty across the entire switch domain. |
| Allocator primary failover | The meta index (which block is where) must survive — without it, TB of KV blocks in CXL memory become unreachable orphans. |

**The meta index — the mapping from `BlockHash` to `BlockHandle` (level + address) — is no longer a soft-state cache. It is a critical data structure whose loss causes TB-scale recomputation.**

### 18.2 Current Meta Index Architecture and Its Gaps

The current meta index is distributed across:

| Component | What It Stores | Durability |
|---|---|---|
| `LocalBlockIndex` (per host, L3) | `HashMap<BlockHash, GlobalBlockAddr>` | **Volatile** — in-process DRAM. Lost on process restart. |
| `LevelGossip` summaries (L3–L5) | Block hash lists or bloom filters per node | **Volatile** — in-memory, rebuilt from gossip. |
| `CxlMemoryManager` allocation bitmap (L4) | Which CXL blocks are allocated/free/quarantined | **Volatile** — in-process DRAM on the allocator primary. Standby has a replica, but both are in-memory. |
| Radix tree in `RadixKvManager` (L2) | Token-prefix → GPU block mapping | **Volatile** — GPU-local, always recomputable from KV data. |

**Gap 1: No persistence.** A host restart loses all index state. The KV data in CXL memory survives (CXL memory is persistent across process restarts if the memory region is not freed), but the index to find it is gone.

**Gap 2: Gossip doesn't scale to millions of entries.** `LevelSummary::block_hashes` carries full hash lists. At 32 bytes per hash × 1M blocks per host = 32 MB per gossip round. Even with bloom filters, this is excessive at 1ms gossip intervals.

**Gap 3: No incremental rebuild.** If the meta index is lost, there is no way to reconstruct it from the KV data in CXL memory without scanning all blocks and re-hashing.

### 18.3 Design: Tiered Meta Index with Persistence and Incremental Gossip

#### 18.3.1 Meta Index Tiers

The meta index itself follows the hierarchy. At each level, the index has different granularity and durability:

```
Index Tier    Granularity              Durability           Purpose
─────────────────────────────────────────────────────────────────────────
L2 Index      Exact: hash → GPU block  Volatile (OK)        GPU blocks are fast to refill
L3 Index      Exact: hash → CXL addr   Persistent (WAL)     CXL data survives restart
L4 Summary    Bloom filter per host    Replicated (gossip)  Route to correct host
L5 Summary    Bloom filter per switch  Replicated (gossip)  Route to correct switch
L6 Index      Exact: hash → remote key Durable (remote DB)  Already persistent
```

**Key insight:** Only L3 needs new persistence — L2 is cheap to rebuild (GPU), L6 is already persistent (remote store), and L4/L5 summaries are rebuilt from L3 via gossip.

#### 18.3.2 L3 Index Persistence: Write-Ahead Log (WAL)

Add a lightweight WAL to `LocalBlockIndex` that journals index mutations to local SSD/NVMe:

```rust
/// Persistent meta index for L3 (Host) block store.
/// Journals all insert/remove operations to a WAL on local NVMe.
/// On restart, replays the WAL to reconstruct the in-memory index.
pub struct PersistentBlockIndex {
    /// In-memory index (same as v7 LocalBlockIndex).
    inner: LocalBlockIndex,
    /// WAL for persistence. Append-only, periodically compacted.
    wal: IndexWal,
}

/// WAL entry types.
#[derive(Serialize, Deserialize)]
enum WalEntry {
    Insert { hash: BlockHash, addr: GlobalBlockAddr, ref_count: u32 },
    Remove { hash: BlockHash },
    /// Periodic checkpoint: full index snapshot.
    /// After a checkpoint, all prior WAL entries can be discarded.
    Checkpoint { entries: Vec<(BlockHash, GlobalBlockAddr, u32)> },
}

/// Write-ahead log backed by local NVMe.
pub struct IndexWal {
    /// Memory-mapped WAL file.
    file: std::fs::File,
    /// Current write position.
    write_pos: u64,
    /// Checkpoint interval (e.g., every 100K mutations or every 60s).
    checkpoint_interval: u64,
    /// Mutations since last checkpoint.
    mutations_since_checkpoint: u64,
}

impl IndexWal {
    /// Append an entry to the WAL. Fsync is batched (group commit).
    pub fn append(&mut self, entry: &WalEntry) {
        let bytes = bincode::serialize(entry).unwrap();
        let len = bytes.len() as u32;
        // Write: [len: u32][entry bytes][crc32: u32]
        self.file.write_all(&len.to_le_bytes()).unwrap();
        self.file.write_all(&bytes).unwrap();
        let crc = crc32fast::hash(&bytes);
        self.file.write_all(&crc.to_le_bytes()).unwrap();
        self.write_pos += 4 + bytes.len() as u64 + 4;
        self.mutations_since_checkpoint += 1;
    }

    /// Replay WAL to reconstruct the in-memory index.
    pub fn replay(&self) -> HashMap<BlockHash, (GlobalBlockAddr, u32)> {
        let mut index = HashMap::new();
        // Read entries from the WAL, applying inserts/removes in order.
        // Start from the last checkpoint if present.
        for entry in self.iter_from_last_checkpoint() {
            match entry {
                WalEntry::Insert { hash, addr, ref_count } => {
                    index.insert(hash, (addr, ref_count));
                }
                WalEntry::Remove { hash } => {
                    index.remove(&hash);
                }
                WalEntry::Checkpoint { entries } => {
                    index.clear();
                    for (hash, addr, rc) in entries {
                        index.insert(hash, (addr, rc));
                    }
                }
            }
        }
        index
    }

    /// Write a checkpoint (full snapshot) and truncate prior entries.
    pub fn checkpoint(&mut self, index: &HashMap<BlockHash, (GlobalBlockAddr, u32)>) {
        let entries: Vec<_> = index.iter()
            .map(|(h, (a, r))| (*h, *a, *r))
            .collect();
        self.append(&WalEntry::Checkpoint { entries });
        // Truncate everything before the checkpoint.
        self.truncate_before_last_checkpoint();
        self.mutations_since_checkpoint = 0;
    }
}
```

**WAL cost analysis:**

| Parameter | Value | Notes |
|---|---|---|
| WAL entry size | ~44 bytes (hash 32B + addr 8B + crc 4B) | Compact binary encoding |
| Insert rate | ~10K–100K blocks/sec (during active prefill) | Worst case: 100K new blocks/sec |
| WAL throughput | ~4.4 MB/s at 100K inserts/sec | Any NVMe can handle this |
| Checkpoint interval | Every 60s or 100K mutations | Keeps replay time < 1s |
| Checkpoint size | ~44 bytes × 1M blocks = 44 MB | One full index snapshot |
| Replay time on restart | < 1s for 1M blocks | Read 44 MB from NVMe |

**Integration with `HostBlockStore`:**

```rust
impl BlockStore for HostBlockStore {
    fn store(&self, blocks: &[KvBlockData]) -> Vec<Result<BlockHandle, BlockError>> {
        // ... existing allocation and CXL write logic ...

        // Journal the insert to WAL for persistence.
        for (block, handle) in blocks.iter().zip(handles.iter()) {
            if let Ok(h) = handle {
                self.persistent_index.wal.append(&WalEntry::Insert {
                    hash: block.meta.hash,
                    addr: GlobalBlockAddr::decode(h.addr),
                    ref_count: 1,
                });
            }
        }

        // ... return results ...
    }

    fn remove(&self, hashes: &[BlockHash]) {
        // ... existing remove logic ...

        // Journal the remove to WAL.
        for h in hashes {
            self.persistent_index.wal.append(&WalEntry::Remove { hash: *h });
        }
    }
}
```

**Startup sequence with WAL:**

```
1. Map CXL memory regions (v6, unchanged).
2. Replay WAL → reconstruct in-memory LocalBlockIndex.
3. Validate: for each index entry, verify the CXL block's hash header
   matches the index entry (detect stale entries from unclean shutdown).
4. Remove stale entries (CXL block was overwritten or corrupted).
5. Resume normal operation. Gossip will propagate the recovered index.
```

#### 18.3.3 Scalable Gossip: Bloom Filters + Incremental Deltas

Replace `LevelSummary::block_hashes: Vec<BlockHash>` with a two-tier approach:

```rust
/// Scalable summary for hierarchical gossip.
/// Replaces the full hash list in LevelSummary.
#[derive(Clone, Serialize, Deserialize)]
pub struct ScalableSummary {
    pub level: HierarchyLevel,
    pub node_id: u32,
    /// Bloom filter covering all blocks at this level.
    /// Size: ~1 MB for 1M blocks at 1% FPR.
    /// Used for routing decisions (probabilistic "does this node have block X?").
    pub bloom: BloomFilter,
    /// Total blocks stored.
    pub block_count: u32,
    /// Free capacity.
    pub free_blocks: u32,
    /// Monotonic version number. Incremented on every mutation.
    pub version: u64,
    /// Incremental delta since last gossip round.
    /// Contains only the hashes added/removed since `prev_version`.
    pub delta: Option<IndexDelta>,
    pub timestamp_ms: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IndexDelta {
    pub prev_version: u64,
    pub added: Vec<BlockHash>,
    pub removed: Vec<BlockHash>,
}
```

**Gossip bandwidth comparison:**

| Approach | Per-Host Per-Round | 8 Hosts × 10 rounds/sec |
|---|---|---|
| v8 current (full hash list, 1M blocks) | 32 MB | 2.56 GB/s — **infeasible** |
| Bloom filter only (1M blocks, 1% FPR) | 1.2 MB | 96 MB/s — feasible but heavy |
| Bloom filter + incremental delta (100 changes/round) | ~4 KB delta | 320 KB/s — **practical** |

**How it works:**
1. Each host maintains a bloom filter covering its blocks. Rebuilt periodically (every checkpoint).
2. Between rebuilds, gossip sends only the incremental delta (added/removed hashes since last round).
3. Receivers apply deltas to their local copy of the bloom filter.
4. The bloom filter is used for routing decisions — false positives cause a cache miss (not correctness failure), and the actual `BlockStore::contains()` check at the target host is exact.

#### 18.3.4 Allocator State Persistence

The `CxlMemoryManager` allocation bitmap should also be persisted:

```rust
/// Persistent allocator state.
/// The allocation bitmap is checkpointed to local NVMe periodically.
pub struct PersistentAllocator {
    inner: CxlMemoryManager,
    /// Allocation bitmap checkpoint file.
    bitmap_path: PathBuf,
    /// Checkpoint interval.
    checkpoint_interval: Duration,
}

impl PersistentAllocator {
    /// On startup: load bitmap from checkpoint, then reconcile with
    /// the WAL-recovered meta index to fix any inconsistencies.
    pub fn recover(bitmap_path: &Path, recovered_index: &HashMap<BlockHash, GlobalBlockAddr>) -> Self {
        let mut bitmap = Self::load_bitmap(bitmap_path);

        // Mark all blocks referenced by the recovered index as allocated.
        for (_, addr) in recovered_index {
            bitmap.mark_allocated(*addr);
        }

        // Any blocks marked allocated in the bitmap but NOT in the index
        // are leaked — free them.
        let leaked = bitmap.find_allocated_but_unreferenced(recovered_index);
        for addr in leaked {
            bitmap.free(addr);
        }

        Self { inner: CxlMemoryManager::from_bitmap(bitmap), .. }
    }
}
```

### 18.4 Cost-Availability Tradeoff Summary

| Protection Level | What You Pay | What You Get | When to Use |
|---|---|---|---|
| **None** (v7/v8 current) | 0 | Lose entire KVCache on restart | KVCache < 100 GB, recomputation is cheap |
| **WAL only** (L3 index persistence) | ~5 MB/s NVMe write, < 1s restart | Survive process restart, recover TB of CXL KVCache | KVCache 100 GB – 1 TB per host |
| **WAL + allocator checkpoint** | +1 MB periodic checkpoint | Survive process restart with consistent allocation state | Multi-host with centralized allocator |
| **WAL + replicated index** | +network bandwidth for index replication | Survive host hardware failure (index exists on peer) | KVCache > 1 TB, recomputation cost > 10 min |

### 18.5 Configuration

```yaml
hierarchy:
  levels:
    - level: host
      meta_index:
        persistence: wal           # none | wal
        wal_path: "/var/lib/llm-server/index.wal"
        checkpoint_interval_secs: 60
        checkpoint_max_mutations: 100000
        validate_on_recovery: true  # verify CXL block hashes on startup
      gossip:
        summary_mode: bloom_delta   # full_hashes | bloom_delta
        bloom_fpr: 0.01            # false positive rate
        bloom_rebuild_interval_secs: 300
```

### 18.6 Impact on v8 Abstractions

| Abstraction | Change |
|---|---|
| `BlockStore` trait | No change — persistence is an implementation detail of `HostBlockStore` |
| `LevelNode` trait | No change |
| `LevelSummary` | **Replaced** by `ScalableSummary` with bloom filter + delta |
| `LevelGossip` | **Modified** to send/receive `ScalableSummary` instead of full hash lists |
| `LocalBlockIndex` | **Wrapped** by `PersistentBlockIndex` |
| `CxlMemoryManager` | **Wrapped** by `PersistentAllocator` |
| New dependency | `crc32fast` (WAL checksums), `bloomfilter` or custom bloom |

---

## 19. Layer-Wise KVCache Management and Compute-IO Overlap

### 19.1 The Problem: Monolithic Blocks Prevent Pipelining

In the current v8 design, a KV block contains **all layers' KV tensors for a token range**:

```
Current KvBlockData layout:
┌──────────────────────────────────────────────┐
│ Block for tokens [0..128], all layers        │
│ ┌──────────┐┌──────────┐     ┌──────────┐   │
│ │ Layer 0   ││ Layer 1   │ ... │ Layer 79  │  │
│ │ K: [h,d]  ││ K: [h,d]  │     │ K: [h,d]  │ │
│ │ V: [h,d]  ││ V: [h,d]  │     │ V: [h,d]  │ │
│ └──────────┘└──────────┘     └──────────┘   │
│ Total: 80 layers × 2 × heads × head_dim     │
│ = 80 × 2 × 128 × 128 × 2B = ~5 MB/block    │
└──────────────────────────────────────────────┘
```

**Problem:** The attention kernel processes layers sequentially — layer 0 first, then layer 1, etc. But the KV block must be fully loaded before any layer can execute. This creates a serial dependency:

```
Without layer-wise management:
  Time: ──────────────────────────────────────────→
  IO:   [===== Load full block (5MB) =====]
  GPU:                                      [L0][L1][L2]...[L79]
                                            ↑ GPU idle during load
```

With layer-wise management, the GPU can start computing attention on layer 0 while layers 1+ are still being loaded:

```
With layer-wise management:
  Time: ──────────────────────────────────────────→
  IO:   [L0][L1][L2][L3]...............[L79]
  GPU:      [L0][L1][L2][L3]...........[L79]
            ↑ overlap: GPU computes L0 while IO loads L1
```

### 19.2 Layer-Aware Block Identity

Extend `BlockHash` and `KvBlockMeta` to include layer information:

```rust
/// Extended block identity that includes layer range.
/// A LayerBlockId uniquely identifies KV data for a specific
/// token prefix AND a specific layer range.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerBlockId {
    /// Hash of the token prefix (same as current BlockHash).
    pub prefix_hash: BlockHash,
    /// Layer range [start, end) that this block covers.
    /// For monolithic blocks: start=0, end=num_layers.
    /// For per-layer blocks: start=layer_idx, end=layer_idx+1.
    /// For layer-group blocks: e.g., start=0, end=8 (8 layers per group).
    pub layer_start: u16,
    pub layer_end: u16,
}

impl LayerBlockId {
    /// Compute a unique hash incorporating both prefix and layer range.
    pub fn block_hash(&self) -> BlockHash {
        let mut hasher = sha2::Sha256::new();
        hasher.update(&self.prefix_hash);
        hasher.update(&self.layer_start.to_le_bytes());
        hasher.update(&self.layer_end.to_le_bytes());
        hasher.finalize().into()
    }

    /// Is this a monolithic (all-layers) block?
    pub fn is_monolithic(&self) -> bool {
        self.layer_start == 0 && self.layer_end == u16::MAX
    }

    /// Number of layers in this block.
    pub fn layer_count(&self) -> u16 {
        self.layer_end - self.layer_start
    }
}
```

### 19.3 Layer Granularity Strategies

The optimal layer granularity depends on the ratio of IO bandwidth to compute throughput:

| Strategy | Block Count per Token Range | Block Size (80-layer model, 128 tokens) | When to Use |
|---|---|---|---|
| **Monolithic** (current) | 1 | ~5 MB | GPU-local HBM (L2). No IO overlap needed. |
| **Per-layer** | 80 | ~64 KB each | CXL load (L3–L5) where fine-grained prefetch matters. |
| **Layer-group** (e.g., 8 layers) | 10 | ~640 KB each | Balance between prefetch granularity and index overhead. |

**The right granularity varies by hierarchy level:**

```
L2 (GPU HBM):     Monolithic — already in HBM, no transfer needed.
L3 (Host DRAM):   Layer-group (4–8 layers) — CXL load latency ~200ns/cacheline,
                   need enough per-group to amortize index overhead.
L4–L5 (CXL pool): Layer-group (8–16 layers) — higher latency, larger groups.
L6 (Remote):      Monolithic — network transfer dominates, prefetch entire block.
```

### 19.4 Layer-Wise Prefetch Engine

The core mechanism: a **prefetch engine** that works with the attention kernel's layer-by-layer execution to load KVCache layers ahead of compute.

```rust
/// Layer-wise prefetch engine.
/// Coordinates KVCache loading from BlockStore with attention kernel execution.
///
/// The engine maintains a sliding window of layer groups:
/// - "computed": layers whose attention has completed
/// - "ready": layers whose KV data is in GPU HBM, ready for attention
/// - "in-flight": layers whose KV data is being transferred to GPU HBM
/// - "pending": layers not yet scheduled for transfer
pub struct LayerPrefetchEngine {
    /// Total layers in the model.
    num_layers: u32,
    /// Layer group size (how many layers per prefetch unit).
    group_size: u32,
    /// Number of layer groups to prefetch ahead of compute.
    prefetch_depth: u32,
    /// The BlockStore hierarchy (to fetch from CXL/DRAM/remote).
    hierarchy_root: Arc<dyn LevelNode>,
    /// CUDA stream for async DMA transfers.
    transfer_stream: CudaStream,
    /// Per-group transfer state.
    group_states: Vec<LayerGroupState>,
}

#[derive(Clone, Debug)]
enum LayerGroupState {
    /// Not yet scheduled for transfer.
    Pending,
    /// Transfer in progress (DMA from CXL/DRAM to GPU HBM).
    InFlight { event: CudaEvent },
    /// Data is in GPU HBM, ready for attention kernel.
    Ready,
    /// Attention has completed for this group.
    Computed,
}

impl LayerPrefetchEngine {
    /// Called before the attention kernel for each layer group.
    /// Ensures the KV data for this group is in GPU HBM.
    /// Returns a pointer to the KV data in GPU HBM.
    ///
    /// Non-blocking: if the data is already Ready, returns immediately.
    /// If InFlight, synchronizes on the CUDA event.
    /// Also kicks off prefetch for the next `prefetch_depth` groups.
    pub fn get_kv_for_layer_group(
        &mut self,
        prefix_hash: &BlockHash,
        group_idx: u32,
    ) -> GpuKvPointer {
        let group = &mut self.group_states[group_idx as usize];

        match group {
            LayerGroupState::Ready => {
                // Already in GPU HBM — return pointer.
            }
            LayerGroupState::InFlight { event } => {
                // Wait for the transfer to complete.
                event.synchronize();
                *group = LayerGroupState::Ready;
            }
            LayerGroupState::Pending => {
                // Synchronous fallback: load now (stall).
                self.sync_load_group(prefix_hash, group_idx);
            }
            LayerGroupState::Computed => {
                panic!("Requesting already-computed layer group");
            }
        }

        // Kick off prefetch for upcoming groups.
        self.prefetch_ahead(prefix_hash, group_idx);

        self.gpu_kv_pointer(group_idx)
    }

    /// Asynchronously prefetch the next N groups.
    fn prefetch_ahead(&mut self, prefix_hash: &BlockHash, current_group: u32) {
        for offset in 1..=self.prefetch_depth {
            let target = current_group + offset;
            if target >= self.num_groups() { break; }

            if let LayerGroupState::Pending = &self.group_states[target as usize] {
                self.async_load_group(prefix_hash, target);
            }
        }
    }

    /// Async load: fetch from BlockStore, DMA to GPU, set InFlight.
    fn async_load_group(&mut self, prefix_hash: &BlockHash, group_idx: u32) {
        let layer_start = group_idx * self.group_size;
        let layer_end = (layer_start + self.group_size).min(self.num_layers);

        let layer_block_id = LayerBlockId {
            prefix_hash: *prefix_hash,
            layer_start: layer_start as u16,
            layer_end: layer_end as u16,
        };

        // Fetch from the hierarchy (will hit L3 CXL, L4, etc.)
        let hash = layer_block_id.block_hash();
        let data = self.fetch_from_hierarchy(&hash);

        // Async DMA to GPU HBM on the transfer stream.
        let event = self.transfer_stream.async_memcpy_h2d(
            &data,
            self.gpu_buffer_for_group(group_idx),
        );

        self.group_states[group_idx as usize] = LayerGroupState::InFlight { event };
    }

    fn num_groups(&self) -> u32 {
        (self.num_layers + self.group_size - 1) / self.group_size
    }
}
```

### 19.5 Integration with the Attention Kernel

The attention kernel must cooperate with the prefetch engine. Two integration patterns:

#### Pattern A: Callback-Driven (Kernel Calls Prefetch Engine)

The attention kernel calls the prefetch engine before each layer group:

```rust
/// Modified forward pass that integrates layer-wise prefetch.
pub fn forward_pass_with_prefetch(
    model: &Model,
    input: &Tensor,
    kv_cache: &mut LayerPrefetchEngine,
    prefix_hash: &BlockHash,
) -> Tensor {
    let mut hidden = input.clone();

    for group_idx in 0..kv_cache.num_groups() {
        // Get KV data for this layer group (may block on prefetch).
        let kv_ptr = kv_cache.get_kv_for_layer_group(prefix_hash, group_idx);

        let layer_start = group_idx * kv_cache.group_size;
        let layer_end = (layer_start + kv_cache.group_size).min(model.num_layers);

        for layer in layer_start..layer_end {
            // Standard attention + FFN, using the prefetched KV data.
            hidden = model.layers[layer as usize].attention(&hidden, &kv_ptr, layer);
            hidden = model.layers[layer as usize].ffn(&hidden);
        }

        kv_cache.mark_group_computed(group_idx);
    }

    hidden
}
```

#### Pattern B: Double-Buffered (Separate Prefetch and Compute Streams)

Use two CUDA streams — one for compute, one for DMA — with event-based synchronization:

```
CUDA Stream 0 (Compute):  [Attn L0-7] ──event→ [Attn L8-15] ──event→ [Attn L16-23] ...
                               ↑                     ↑                      ↑
                           wait(E_0)              wait(E_1)             wait(E_2)

CUDA Stream 1 (DMA):     [DMA L0-7]  [DMA L8-15]  [DMA L16-23]  [DMA L24-31] ...
                              ↓            ↓             ↓             ↓
                          record(E_0)  record(E_1)   record(E_2)  record(E_3)
```

```rust
/// Double-buffered prefetch with two CUDA streams.
pub struct DoubleBufferPrefetch {
    /// Compute stream (runs attention kernels).
    compute_stream: CudaStream,
    /// Transfer stream (runs async DMA).
    transfer_stream: CudaStream,
    /// Two GPU buffers for double-buffering.
    buffers: [GpuBuffer; 2],
    /// Events for stream synchronization.
    transfer_events: Vec<CudaEvent>,
}

impl DoubleBufferPrefetch {
    pub fn execute_pipelined(
        &mut self,
        model: &Model,
        input: &Tensor,
        prefix_hash: &BlockHash,
        hierarchy: &dyn LevelNode,
    ) -> Tensor {
        let num_groups = self.num_groups(model.num_layers);
        let mut hidden = input.clone();

        // Kick off first DMA before any compute.
        self.start_dma(0, prefix_hash, hierarchy);

        for g in 0..num_groups {
            let buf_idx = g % 2;

            // Wait for this group's DMA to complete.
            self.compute_stream.wait_event(&self.transfer_events[g]);

            // Start DMA for group g+2 (double buffer lookahead).
            if g + 2 < num_groups {
                self.start_dma(g + 2, prefix_hash, hierarchy);
            } else if g + 1 < num_groups {
                // Ensure next group is also started.
                self.start_dma(g + 1, prefix_hash, hierarchy);
            }

            // Compute attention for this group's layers.
            let layer_start = g * self.group_size;
            let layer_end = ((g + 1) * self.group_size).min(model.num_layers);
            for layer in layer_start..layer_end {
                hidden = model.layers[layer].attention_with_external_kv(
                    &hidden,
                    &self.buffers[buf_idx],
                    layer - layer_start, // offset within the buffer
                    &self.compute_stream,
                );
                hidden = model.layers[layer].ffn(&hidden, &self.compute_stream);
            }
        }

        hidden
    }

    fn start_dma(&mut self, group_idx: usize, prefix_hash: &BlockHash, hierarchy: &dyn LevelNode) {
        let buf_idx = group_idx % 2;
        let layer_block_id = LayerBlockId {
            prefix_hash: *prefix_hash,
            layer_start: (group_idx * self.group_size) as u16,
            layer_end: ((group_idx + 1) * self.group_size) as u16,
        };
        let hash = layer_block_id.block_hash();

        // Fetch from hierarchy (host DRAM or CXL).
        let data = find_and_fetch_block(hierarchy, &hash);

        // Async DMA on transfer stream.
        self.transfer_stream.async_memcpy_h2d(&data, &self.buffers[buf_idx]);
        self.transfer_stream.record_event(&self.transfer_events[group_idx]);
    }
}
```

### 19.6 Layer-Wise BlockStore Changes

The `BlockStore` trait itself does not change — `LayerBlockId::block_hash()` produces a standard `BlockHash`. The layer granularity is handled at the storage and retrieval level:

```rust
/// Layer-aware block storage.
/// Splits a monolithic KvBlockData into layer-group blocks for L3+ storage.
/// Reassembles them for L2 (GPU) if needed.
pub struct LayerAwareBlockStore {
    inner: Box<dyn BlockStore>,
    /// Layer group size for this level.
    group_size: u32,
    /// Total model layers.
    num_layers: u32,
}

impl LayerAwareBlockStore {
    /// Store a monolithic block as multiple layer-group blocks.
    pub fn store_layered(
        &self,
        prefix_hash: &BlockHash,
        full_kv_data: &[u8],
    ) -> Vec<Result<BlockHandle, BlockError>> {
        let layer_size = full_kv_data.len() / self.num_layers as usize;
        let mut results = Vec::new();

        for group_start in (0..self.num_layers).step_by(self.group_size as usize) {
            let group_end = (group_start + self.group_size).min(self.num_layers);
            let layer_id = LayerBlockId {
                prefix_hash: *prefix_hash,
                layer_start: group_start as u16,
                layer_end: group_end as u16,
            };

            let data_start = group_start as usize * layer_size;
            let data_end = group_end as usize * layer_size;
            let block = KvBlockData {
                meta: KvBlockMeta {
                    hash: layer_id.block_hash(),
                    layer_start: group_start as u16,
                    layer_end: group_end as u16,
                    ..Default::default()
                },
                data: full_kv_data[data_start..data_end].to_vec(),
            };

            results.extend(self.inner.store(&[block]));
        }

        results
    }

    /// Fetch a specific layer group.
    pub fn fetch_layer_group(
        &self,
        prefix_hash: &BlockHash,
        layer_start: u16,
        layer_end: u16,
    ) -> Option<KvBlockData> {
        let layer_id = LayerBlockId {
            prefix_hash: *prefix_hash,
            layer_start,
            layer_end,
        };
        let hash = layer_id.block_hash();
        self.inner.fetch(&[hash]).into_iter().next().flatten()
    }
}
```

### 19.7 Performance Analysis: Compute-IO Overlap

For a 70B model (80 layers, GQA 8 KV heads, head_dim=128, FP16):

| Parameter | Value |
|---|---|
| KV per layer per token | 2 × 8 × 128 × 2B = 4 KB |
| KV per layer-group (8 layers, 128 tokens) | 8 × 128 × 4 KB = 4 MB |
| CXL read bandwidth | ~32 GB/s (CXL 2.0 x16) |
| Time to read 1 layer-group from CXL | 4 MB / 32 GB/s = **125 μs** |
| Attention compute per layer-group (8 layers, 128 tokens, batch 1) | ~8 × 50 μs = **400 μs** |
| Overlap ratio | 125 μs / 400 μs = **31%** — IO fully hidden within compute |

```
Without overlap (monolithic, 80 layers):
  IO:  [====== 1.25 ms (load all) ======]
  GPU:                                    [==== 4 ms (all layers) ====]
  Total: 5.25 ms

With overlap (8-layer groups, 10 groups):
  IO:  [G0][G1][G2]...[G9]  (each 125μs, total 1.25 ms)
  GPU:     [G0][G1][G2]...[G9]  (each 400μs, total 4 ms)
  Total: 4.125 ms  (IO hidden behind compute, except first group)
  Speedup: 5.25 / 4.125 = 1.27× for prefill from CXL
```

For larger batch sizes, compute time per group increases while IO time stays constant — the overlap becomes even more effective, approaching full IO hiding.

### 19.8 Interaction with Eviction and Routing

Layer-wise management affects eviction and routing:

**Eviction:** Layer-group blocks can be evicted independently. Cold layers (layers rarely accessed — e.g., early layers that are common across many prefixes) can stay at higher hierarchy levels while hot layers are promoted. This enables **layer-heterogeneous caching**: keep layers 0-7 (common prefix layers) at L4 (CXL pool, shared across hosts) while layers 60-79 (sequence-specific) stay at L2 (GPU).

**Routing:** The routing decision can consider per-layer hit rates. If a host has layers 0-39 cached but not 40-79, and another host has all 80 layers, the router should prefer the latter for a full-sequence request.

```rust
/// Extended routing score that considers layer coverage.
fn score_child_layer_aware(
    child: &dyn LevelNode,
    prefix_hash: &BlockHash,
    num_layers: u32,
    group_size: u32,
    params: &RoutingParams,
) -> f64 {
    let mut covered_groups = 0u32;
    let total_groups = (num_layers + group_size - 1) / group_size;

    for g in (0..num_layers).step_by(group_size as usize) {
        let layer_id = LayerBlockId {
            prefix_hash: *prefix_hash,
            layer_start: g as u16,
            layer_end: (g + group_size).min(num_layers) as u16,
        };
        let hash = layer_id.block_hash();
        if child.store().contains(&[hash])[0] {
            covered_groups += 1;
        }
    }

    let coverage = covered_groups as f64 / total_groups as f64;
    coverage * params.hit_weight
    // ... plus load and capacity scores
}
```

### 19.9 Configuration

```yaml
hierarchy:
  layer_management:
    enabled: true
    num_layers: 80                # model-specific
    # Per-level layer group sizes.
    # Larger groups = less index overhead, less prefetch granularity.
    group_sizes:
      chip: 0                     # 0 = monolithic (all layers in one block on GPU)
      host: 8                     # 8 layers per group in host DRAM/CXL
      switch_domain: 16           # 16 layers per group in CXL pool
      fabric: 16
      global: 0                   # 0 = monolithic for remote store
    prefetch:
      depth: 2                    # prefetch 2 groups ahead
      double_buffer: true         # use double-buffered DMA
      compute_stream_priority: high
      transfer_stream_priority: low
```

### 19.10 Impact on v8 Abstractions

| Abstraction | Change |
|---|---|
| `BlockHash` / `KvBlockMeta` | **Extended** with `layer_start`/`layer_end` fields. Old monolithic blocks have `layer_start=0, layer_end=MAX`. |
| `BlockStore` trait | **No change** — layer-group blocks are just blocks with different hashes. |
| `LevelNode` trait | **No change** — routing uses `BlockStore::contains()` which works with any hash. |
| `LayerPrefetchEngine` | **New** — coordinates layer-wise prefetch with attention compute. |
| `DoubleBufferPrefetch` | **New** — double-buffered DMA for pipelined execution. |
| `LayerAwareBlockStore` | **New** — wrapper that splits/reassembles layer-group blocks. |
| Forward pass (`executor/`) | **Modified** — integrates with `LayerPrefetchEngine` for CXL-sourced KV. |
| `RoutingParams` | **Extended** — optional layer-coverage scoring. |

### 19.11 Implementation Phases

**Phase 13a — Layer Block Identity (0.5 weeks):**
- [ ] `LayerBlockId` struct and `block_hash()` method
- [ ] Extend `KvBlockMeta` with `layer_start`/`layer_end`
- [ ] `LayerAwareBlockStore` wrapper for split/reassemble
- [ ] Tests: round-trip layer-group blocks through `BlockStore`

**Phase 13b — Prefetch Engine (1 week):**
- [ ] `LayerPrefetchEngine` with `get_kv_for_layer_group()`
- [ ] `DoubleBufferPrefetch` with dual CUDA streams
- [ ] Integration with forward pass
- [ ] Benchmarks: prefill latency with/without layer-wise prefetch

**Phase 13c — Layer-Aware Routing (0.5 weeks):**
- [ ] `score_child_layer_aware()` with layer coverage scoring
- [ ] Layer-heterogeneous eviction policy
- [ ] Tests: routing prefers hosts with higher layer coverage

---

## Key Design Decisions — Rationale

**Traits over type erasure:** `BlockStore` and `LevelNode` are trait objects (`dyn BlockStore`, `dyn LevelNode`). This adds vtable overhead (~1ns per call) but enables the recursive tree to hold heterogeneous nodes (GPU, CXL, remote). At 1ns per vtable lookup vs. 10ms per inference step, this is unmeasurable.

**Wrap, don't rewrite:** v8 wraps v7's components (`LocalBlockIndex`, `CxlMemoryManager`, `RadixKvManager`) behind traits rather than rewriting them. This preserves v7's correctness guarantees while adding the recursive structure. The implementations are battle-tested; the traits add uniformity without changing behavior.

**Level collapsing over level skipping:** When a hierarchy level has only one node, v8 collapses it (passes children through directly) rather than skipping it in the code. This means the collapsed level's `LevelNode` still exists — it just delegates immediately to its single child. This keeps the code structure consistent regardless of topology complexity.

**Uniform metrics enable uniform monitoring:** With `LevelMetrics[L]` having the same schema at every level, a single Grafana dashboard can show all levels side-by-side. A single Prometheus alert rule can fire on any level's `integrity_failures`. This is a practical benefit of recursive decomposition that goes beyond code aesthetics.

**Eviction cascades up, promotion descends down:** This directional consistency (eviction = toward root = toward higher latency/higher capacity; promotion = toward leaves = toward lower latency/lower capacity) mirrors the Linqu bandwidth gradient: as level number increases, latency increases and bandwidth decreases. Eviction moves data toward cheaper, larger storage; promotion moves data toward faster, smaller storage.

---

*v8 inherits v7's corruption-resilient CXL architecture and restructures it around the Linqu hierarchical symmetry principle. The key change is not what the system does — it is how the system is organized. Every level now implements the same traits, uses the same metrics, follows the same routing/eviction/integrity patterns, and is configured with the same schema. This makes the system easier to extend (add a level), easier to monitor (uniform metrics), and easier to reason about (same patterns at every level). The performance characteristics are identical to v7 — the recursive structure is a zero-cost abstraction over the existing level-specific implementations. §18 adds meta index persistence (WAL + scalable gossip via bloom filters) to make TB-scale KVCache recoverable without recomputation. §19 adds layer-wise KVCache management with double-buffered prefetch to overlap compute and IO, hiding CXL transfer latency behind attention computation.*
