# RustLLM_serve

🦀 Lightweight HTTP server in Rust to run LLM models on edge without requiring expensive compute and dependencies.

## ✨ Features

- **Minimal Dependencies**: Built with Candle ML framework - no PyTorch or heavy ML frameworks required
- **HTTP API**: Simple REST API with Axum for easy integration
- **CUDA Support**: Optional GPU acceleration (just uncomment in `Cargo.toml`)
- **Temperature & Seed Control**: Fine-tune generation behavior
- **Auto Model Download**: Automatically fetches models from Hugging Face Hub
- **CORS Enabled**: Ready for web application integration

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- (Optional) CUDA 11.8+ for GPU acceleration

### Installation

```bash
git clone https://github.com/yourusername/RustLLM_serve.git
cd RustLLM_serve
cargo build --release
```

### Run the Server

```bash
cargo run --release
```

The server will start on `http://0.0.0.0:3000` and automatically download the TinyLlama model on first run.

## 📡 API Endpoints

### Health Check

Check if the server is running and which model is loaded.

**Endpoint:** `GET /health`

**Example:**
```bash
curl http://localhost:3000/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}
```

### Text Generation (Inference)

Generate text completion from a prompt.

**Endpoint:** `POST /inference`

**Request Body:**
```json
{
  "prompt": "Hello, my name is",
  "max_tokens": 50,
  "temperature": 0.8,
  "seed": 42
}
```

**Parameters:**
- `prompt` (required): Input text to complete
- `max_tokens` (optional): Maximum tokens to generate (default: 50, max: 1024)
- `temperature` (optional): Sampling temperature 0.01-100.0 (default: 1.0)
  - Lower values → more deterministic
  - Higher values → more creative/random
- `seed` (optional): Random seed for reproducibility (default: 42)

**Example:**
```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "generated_text": "The future of AI is bright and full of possibilities..."
}
```

## 🔧 Configuration

### Enable CUDA (GPU Acceleration)

Edit `Cargo.toml` and uncomment the CUDA features:

```toml
# Comment these lines:
# candle-core = "0.9.2"
# candle-nn = "0.9.2"

# Uncomment these lines:
candle-core = { version = "0.9.2", features = ["cuda"] }
candle-nn = { version = "0.9.2", features = ["cuda"] }
```

### Change Model

In `src/main.rs`, modify the model repository:

```rust
let repo_id = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());
// Change to any compatible Llama-architecture model from Hugging Face
```

### Change Server Port

In `src/main.rs`:

```rust
let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
// Change port as needed
```

## 🏗️ Project Structure

```
RustLLM_serve/
├── src/
│   ├── main.rs              # Server initialization and model loading
│   ├── config.rs            # Model configuration structures
│   ├── api/
│   │   ├── handlers.rs      # HTTP request handlers
│   │   ├── models.rs        # Request/Response models
│   │   └── server.rs        # Router configuration
│   └── llm/
│       ├── models.rs        # LLM model trait and implementations
│       ├── inference.rs     # Text generation logic
│       ├── decoder.rs       # Transformer decoder
│       ├── attention.rs     # Self-attention mechanism
│       ├── embedding.rs     # Token embeddings
│       └── ...              # Other model components
├── Cargo.toml
└── README.md
```

## 🧪 Testing

Test the health endpoint:
```bash
curl http://localhost:3000/health
```

Test inference:
```bash
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 50}'
```

## 🔍 Logging

The server uses `tracing` for structured logging. Logs include:
- Model loading progress
- Inference requests and responses
- HTTP request traces

Set log level via environment variable:
```bash
RUST_LOG=debug cargo run
```

## ⚡ Performance Tips

1. **Use Release Mode**: Always run with `--release` for production
2. **Enable CUDA**: GPU acceleration provides 10-100x speedup
3. **Adjust max_tokens**: Lower values = faster responses
4. **Batch Requests**: The server handles concurrent requests efficiently

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Built with [Candle](https://github.com/huggingface/candle) - Hugging Face's Rust ML framework
- Uses [Axum](https://github.com/tokio-rs/axum) for HTTP server
- Models from [Hugging Face Hub](https://huggingface.co/)
