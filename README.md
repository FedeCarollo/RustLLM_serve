# Custom LLM Inference Engine (Rust)

A high-performance LLM inference engine built from scratch in Rust. This project was developed to explore and master the deep mechanics of Large Language Models (specifically LLaMA architectures like TinyLlama) by bypassing high-level abstractions and working directly with bare-metal tensor primitives.

## 🚀 Key Features

*   **From-Scratch Architecture:** Manually implemented the full forward pass, including Rotary Positional Embeddings (RoPE), RMSNorm, and Attention mechanisms using HuggingFace's `candle-core`.
*   **High-Performance KV Cache:** Engineered a pre-allocated Key-Value Cache that performs zero-copy, in-place tensor updates (`slice_set`). This eliminates memory reallocation bottlenecks during auto-regressive decoding, achieving up to a **14x speedup** on long sequences (tested on NVIDIA RTX 5060 / CUDA).
*   **Thread-Safe Concurrency:** Integrated the model into a fully concurrent backend. Utilizes `RwLock` for internal cache state and `tokio::sync::Mutex` to serialize generation requests, ensuring memory safety across concurrent API calls.
*   **REST API:** Wrapped the inference engine in a fast, asynchronous Axum web server to easily serve text generation requests.

## 🛠️ Prerequisites

*   Rust (`cargo`)
*   CUDA Toolkit (for GPU acceleration)

*Note: If you are using a very modern GPU architecture (e.g., Blackwell) with an older CUDA Toolkit, ensure `CUDA_COMPUTE_CAP` is correctly set in your environment or `.cargo/config.toml` (e.g., `CUDA_COMPUTE_CAP="89"`).*

## 🏎️ Quick Start

1. **Run the server:**
   ```bash
   cargo run --release
   ```
   The server will automatically download the TinyLlama-1.1B weights via HuggingFace Hub on the first run and load them into GPU memory.

2. **Generate Text:**
   Send a POST request to the `/inference` endpoint:
   ```bash
   curl -X POST http://127.0.0.1:3000/inference \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, my name is Rust and I", "max_tokens": 50, "temperature": 0.7}'
   ```

3. **Check Health:**
   ```bash
   curl http://127.0.0.1:3000/health
   ```

## 📊 Benchmark

Included is an isolated benchmark to test the speedup provided by the custom KV Cache implementation.

```bash
cargo run --release --example benchmark_cache
```

**Results (50 tokens) in local:**
*   Without KV Cache: ~10.0s (195ms/token)
*   With KV Cache: ~0.68s (13ms/token)
*   **Speedup:** ~14.2x faster 🚀
