// The binary inherits all library deps but doesn't directly use them in its own
// compilation unit. Suppress unused-crate-dependencies for the binary target.
#![allow(unused_crate_dependencies)]
// Allow println! for the startup banner.
#![allow(clippy::print_stdout)]

use std::sync::Arc;

use clap::Parser;
use rust_llm_server::distributed::DistributedConfig;
use rust_llm_server::engine::engine::Engine;
use rust_llm_server::engine::engine::GenerationConfig;
use rust_llm_server::model::config::Qwen3Config;
use rust_llm_server::model::network::Qwen3Model;
use rust_llm_server::model::parallel::ParallelConfig;
use rust_llm_server::model::quantize::QuantConfig;
use rust_llm_server::model::weights::{self, SafetensorsLoader};
use rust_llm_server::scheduler::Qwen3Tokenizer;
use rust_llm_server::server::{AppState, serve};

#[cfg(all(feature = "ascend", feature = "hccl"))]
use rust_llm_server::distributed::process_group;
#[cfg(feature = "ascend")]
use rust_llm_server::ops::ascend::AscendComputeOps;
#[cfg(all(feature = "ascend", feature = "hccl"))]
use rust_llm_server::ops::ascend_comm::AscendCommOps;

/// Rust LLM inference server for Qwen3 models.
#[derive(Parser, Debug)]
#[command(name = "rust-llm-server")]
#[command(about = "LLM inference server framework for Qwen3 models")]
struct Cli {
    /// Path to model config.json (if omitted, uses Qwen3-8B defaults).
    #[arg(long)]
    config: Option<String>,

    /// Model variant shortcut: "0.6b", "4b", "8b".
    #[arg(long, default_value = "8b")]
    model: String,

    /// Server port.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Tensor parallelism degree.
    #[arg(long, default_value_t = 1)]
    tp: usize,

    /// Pipeline parallelism degree.
    #[arg(long, default_value_t = 1)]
    pp: usize,

    /// This device's TP rank.
    #[arg(long, default_value_t = 0)]
    tp_rank: usize,

    /// This device's PP rank.
    #[arg(long, default_value_t = 0)]
    pp_rank: usize,

    /// Data parallelism degree (number of independent replicas).
    #[arg(long, default_value_t = 1)]
    dp: usize,

    /// Quantization: "none", "int8", "awq-int4".
    #[arg(long, default_value = "none")]
    quant: String,

    /// Backend: "stub" (no-op) or "ascend" (Ascend NPU via CANN).
    /// Defaults to "ascend" when built with --features ascend, "stub" otherwise.
    #[cfg_attr(not(feature = "ascend"), arg(long, default_value = "stub"))]
    #[cfg_attr(feature = "ascend", arg(long, default_value = "ascend"))]
    backend: String,

    /// Ascend NPU device ID (only used with --backend ascend).
    /// If not specified, reads TASK_DEVICE, then ASCEND_DEVICE_ID env vars (default: 0).
    #[arg(long)]
    device_id: Option<i32>,

    /// Path to model weights directory (containing *.safetensors files).
    /// If not specified, the server runs with uninitialized weights (shape-only).
    #[arg(long)]
    weights: Option<String>,

    /// One-shot generation: prompt text (no HTTP server, just generate and print).
    /// Use with --weights and optionally --model / --config.
    #[arg(long)]
    prompt: Option<String>,

    /// Maximum new tokens for one-shot generation (default: 128).
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn core::error::Error>> {
    // Inject CANN HCCL workaround for AICPU exception 507018 on AllReduce/Broadcast
    // SAFETY: Called before any threads are spawned; single-threaded startup.
    unsafe {
        std::env::set_var("HCCL_OP_EXPANSION_MODE", "AIV");
        std::env::set_var("HCCL_BUFFSIZE", "512");
    }

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rust_llm_server=info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Load or create config
    // Priority: --config > auto-detect from --weights dir > hardcoded defaults
    let config = if let Some(config_path) = &cli.config {
        tracing::info!("Loading config from {config_path}");
        Qwen3Config::from_json(config_path)?
    } else if let Some(ref weights_dir) = cli.weights {
        // Auto-detect config.json in weights directory
        let auto_config = std::path::Path::new(weights_dir).join("config.json");
        if auto_config.exists() {
            tracing::info!(
                "Auto-detected config from weights dir: {}",
                auto_config.display()
            );
            Qwen3Config::from_json(auto_config.to_str().unwrap())?
        } else {
            tracing::info!(
                "No config.json in weights dir, using --model {} defaults",
                cli.model
            );
            match cli.model.as_str() {
                "0.6b" => Qwen3Config::qwen3_0_6b(),
                "4b" => Qwen3Config::qwen3_4b(),
                "8b" => Qwen3Config::qwen3_8b(),
                "14b" => Qwen3Config::qwen3_14b(),
                other => {
                    return Err(format!(
                        "Unknown model variant: {other}. Use 0.6b, 4b, 8b, or 14b."
                    )
                    .into());
                }
            }
        }
    } else {
        match cli.model.as_str() {
            "0.6b" => {
                tracing::info!("Using Qwen3-0.6B default config");
                Qwen3Config::qwen3_0_6b()
            }
            "4b" => {
                tracing::info!("Using Qwen3-4B default config");
                Qwen3Config::qwen3_4b()
            }
            "8b" => {
                tracing::info!("Using Qwen3-8B default config");
                Qwen3Config::qwen3_8b()
            }
            "14b" => {
                tracing::info!("Using Qwen3-14B default config");
                Qwen3Config::qwen3_14b()
            }
            other => {
                return Err(
                    format!("Unknown model variant: {other}. Use 0.6b, 4b, 8b, or 14b.").into(),
                );
            }
        }
    };

    // Build parallel config — use env vars (RANK, WORLD_SIZE) if set,
    // otherwise fall back to explicit --tp-rank / --pp-rank CLI args.
    let distributed = if std::env::var("RANK").is_ok() && (cli.tp > 1 || cli.pp > 1 || cli.dp > 1) {
        match DistributedConfig::from_env(cli.tp, cli.pp, cli.dp) {
            Ok(dist) => {
                tracing::info!(
                    "Distributed mode: world_rank={}/{}, tp_rank={}, pp_rank={}, dp_rank={}, device={}",
                    dist.world_rank,
                    dist.world_size,
                    dist.tp_rank,
                    dist.pp_rank,
                    dist.dp_rank,
                    dist.device_id(),
                );
                Some(dist)
            }
            Err(e) => {
                tracing::warn!("Failed to init distributed from env: {}", e);
                None
            }
        }
    } else {
        None
    };

    let parallel = if let Some(ref dist) = distributed {
        dist.to_parallel_config()
    } else {
        ParallelConfig {
            tp_size: cli.tp,
            pp_size: cli.pp,
            tp_rank: cli.tp_rank,
            pp_rank: cli.pp_rank,
            dp_size: cli.dp,
            dp_rank: 0,
        }
    };

    // Build quant config
    let quant = match cli.quant.as_str() {
        "none" => QuantConfig::none(),
        "int8" => QuantConfig::int8_per_tensor(),
        "awq-int4" => QuantConfig::awq_int4(128),
        other => {
            return Err(format!("Unknown quant: {other}. Use none, int8, or awq-int4.").into());
        }
    };

    // Select backend — for Ascend, create AscendComputeOps once here and reuse it
    // below for Engine::new_ascend(). OpsBundle::ascend() would create a second
    // Device::init() call which CANN rejects (error 507033: aclrtSetDevice twice).
    #[cfg(feature = "ascend")]
    let ascend_ops_init: Option<AscendComputeOps> = if cli.backend == "ascend" {
        // --device-id takes priority; fall back to LOCAL_RANK from distributed config
        let device_id = cli
            .device_id
            .or_else(|| distributed.as_ref().map(|d| d.device_id()));
        tracing::info!("Using ASCEND NPU backend");
        let ops = AscendComputeOps::new(device_id)
            .map_err(|e| format!("Failed to init Ascend backend: {}", e))?;
        Some(ops)
    } else {
        None
    };

    let backend_label = match cli.backend.as_str() {
        "stub" => {
            tracing::info!("Using STUB backend (no-op operators)");
            "STUB (no-op)"
        }
        #[cfg(feature = "ascend")]
        "ascend" => "ASCEND NPU (CANN)",
        #[cfg(not(feature = "ascend"))]
        "ascend" => {
            return Err(
                "Ascend backend requested but not compiled. Rebuild with: cargo build --features ascend".into()
            );
        }
        other => return Err(format!("Unknown backend: {other}. Use stub or ascend.").into()),
    };

    // Build model (sharded if using TP/PP)
    let mut model = if parallel.is_tp() || parallel.is_pp() {
        tracing::info!(
            "Building sharded model (tp={}, pp={})",
            parallel.tp_size,
            parallel.pp_size
        );
        Qwen3Model::new_sharded(config, &parallel)
    } else {
        Qwen3Model::new(config)
    };
    tracing::info!(
        "Model: {} layers, {}B parameters",
        model.num_layers(),
        model.param_count() as f64 / 1e9
    );

    // Load weights if path is provided
    if let Some(weights_dir) = &cli.weights {
        let loader = SafetensorsLoader::from_dir(std::path::Path::new(weights_dir))?;
        if parallel.is_tp() {
            weights::load_weights_sharded(&mut model, &loader, &parallel)?;
        } else {
            weights::load_weights(&mut model, &loader)?;
        }

        // Upload to device if using ascend backend
        #[cfg(feature = "ascend")]
        if cli.backend == "ascend" {
            // Reuse the already-initialized stream from ascend_ops
            if let Some(ref aops) = ascend_ops_init {
                weights::upload_weights_to_device(&mut model, aops.stream())?;
            }
        }
    } else {
        tracing::warn!("No --weights specified, running with uninitialized weights");
    }

    // Create engine with compiled execution plan.
    // `mut` is only needed when `hccl` is enabled (for distributed init below).
    #[cfg(feature = "ascend")]
    #[cfg_attr(not(feature = "hccl"), allow(unused_mut))]
    let mut engine = if let Some(ascend_ops) = ascend_ops_init {
        Engine::new_ascend(model, ascend_ops, parallel.clone(), quant)
    } else {
        Engine::new(model, parallel.clone(), quant)
    };
    #[cfg(not(feature = "ascend"))]
    let engine = Engine::new(model, parallel.clone(), quant);
    tracing::info!("Engine: {}", engine.model_info());

    // Initialize HCCL communicators for distributed execution
    #[cfg(all(feature = "ascend", feature = "hccl"))]
    if let Some(ref dist) = distributed
        && !dist.is_single()
        && cli.backend == "ascend"
    {
        let root_info_dir = std::path::Path::new("/tmp/hccl_root_info");
        let groups = process_group::init_process_groups(dist, root_info_dir)
            .map_err(|e| format!("Failed to init HCCL process groups: {}", e))?;

        // Use the compute stream for HCCL ops (like vLLM uses current_stream()).
        // Using a separate comm stream causes AICPU exceptions (507018)
        // because HCCL internally requires stream/context alignment.
        let comm_stream = engine
            .compute_stream()
            .expect("compute stream required for HCCL comm ops");
        let comm_ops = AscendCommOps::new(groups.tp_comm, groups.pp_comm, comm_stream);
        engine.set_comm_ops(comm_ops);

        // Clean up root info files after all ranks have initialized
        process_group::cleanup_root_info(root_info_dir);
        tracing::info!("HCCL communicators initialized for distributed execution");
    }

    // Load tokenizer from weights directory (shared by one-shot and HTTP modes)
    let tokenizer = if let Some(ref weights_dir) = cli.weights {
        let tokenizer_path = std::path::Path::new(weights_dir).join("tokenizer.json");
        tracing::info!("Loading tokenizer from: {}", tokenizer_path.display());
        Qwen3Tokenizer::from_file(tokenizer_path.to_str().unwrap())
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?
    } else {
        return Err("--weights is required to load tokenizer.json".into());
    };

    // ─── One-shot generation mode ──────────────────────────────────────
    // If --prompt is provided, run a single generation and print the result,
    // then exit. No HTTP server is started in this mode.
    if let Some(prompt_text) = &cli.prompt {
        tracing::info!(
            "One-shot generation: prompt={:?}, max_new_tokens={}",
            prompt_text,
            cli.max_new_tokens
        );

        let gen_config = GenerationConfig {
            max_new_tokens: cli.max_new_tokens,
            ..Default::default()
        };

        let prompt_ids = tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err("Empty prompt after tokenization".into());
        }
        tracing::info!("Prompt tokens: {}", prompt_ids.len());

        let result = engine.generate(&prompt_ids, &gen_config);

        let text = tokenizer.decode(&result.token_ids);
        let output = serde_json::json!({
            "prompt": prompt_text,
            "generated": text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());

        return Ok(());
    }

    // Create app state
    let state = Arc::new(AppState { engine, tokenizer });

    // Determine which rank serves HTTP.
    // In a TP/PP setup only one rank per DP replica should bind the port:
    //   - tp_rank == 0  (only one rank per TP group handles the HTTP interface)
    //   - pp_rank == pp_size - 1  (last PP stage produces the output logits)
    // All other TP/PP ranks are "worker" ranks that receive work via HCCL and
    // must NOT try to bind the same port.
    let is_primary = if let Some(ref dist) = distributed {
        dist.tp_rank == 0 && dist.pp_rank == cli.pp.saturating_sub(1)
    } else {
        // Single-process mode — always the primary
        parallel.tp_rank == 0 && parallel.pp_rank == parallel.pp_size.saturating_sub(1)
    };

    // DP: offset port by dp_rank so each DP replica gets its own port
    let serve_port = if let Some(ref dist) = distributed {
        cli.port + dist.dp_rank as u16
    } else {
        cli.port
    };

    if !is_primary {
        // Worker rank: does not serve HTTP.
        // With HCCL enabled, enters a blocking loop that:
        //   1. Receives input_ids and positions via HcclBroadcast from rank 0
        //   2. Runs execute_paged() (which triggers HCCL AllReduce in lockstep with primary)
        //   3. Discards the output and repeats
        // This is the vLLM worker_busy_loop pattern — NPU-level sync only, no CPU mutex.
        tracing::info!(
            "Worker rank (tp_rank={}, pp_rank={}): entering HCCL broadcast worker loop",
            parallel.tp_rank,
            parallel.pp_rank
        );

        #[cfg(all(feature = "ascend", feature = "hccl"))]
        {
            // Take ownership of Engine out of the Arc<AppState>.
            // At this point exactly one Arc reference exists (no HTTP server started yet).
            let app_state = Arc::try_unwrap(state).unwrap_or_else(|_| {
                panic!("Worker: Arc<AppState> had unexpected additional references")
            });
            let engine = app_state.engine;
            let handle = std::thread::spawn(move || engine.run_worker_loop());
            handle.join().expect("Worker loop thread panicked");
        }
        #[cfg(not(all(feature = "ascend", feature = "hccl")))]
        {
            // Non-HCCL build: park the thread (no computation to mirror)
            core::future::pending::<()>().await;
        }
        return Ok(());
    }

    // Start server (primary rank only)
    println!("╔════════════════════════════════════════════════╗");
    println!("║  Rust LLM Server — Qwen3 (Compiled Plan)      ║");
    println!("║  {}", state.engine.model_info());
    println!("║  Operators: {:38}║", backend_label);
    println!("║  Endpoints:                                    ║");
    println!("║    GET  /health          POST /v1/completions  ║");
    println!("║    GET  /v1/models                             ║");
    println!("║  Port: {:41}║", serve_port);
    println!("╚════════════════════════════════════════════════╝");

    serve(state, serve_port).await?;
    Ok(())
}
