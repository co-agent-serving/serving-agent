//! `rust_llm_server` — Rust LLM inference server framework for Qwen3 models.
//!
//! This library provides the core inference engine, model definitions,
//! weight loading, operator backends, scheduling, and HTTP server.
//! The binary entry point lives in `main.rs`.

// Suppress unused-crate-dependencies for deps used only in the binary.
use clap as _;
use half as _;
use tracing_subscriber as _;

pub mod distributed;
pub mod engine;
pub mod model;
pub mod ops;
pub mod scheduler;
pub mod server;
