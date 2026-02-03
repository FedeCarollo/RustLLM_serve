use std::fs::File;
use std::sync::Arc;

use anyhow::Error;
use candle_core::{Device, safetensors::MmapedSafetensors};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use std::path::Path;

mod config;
mod llm;
mod api;

pub struct AppState {
    pub model: Box<dyn llm::models::Model + Send + Sync>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub model_name: String,
    pub eos_token_id: i64,
}

#[tokio::main]
async fn main() -> Result<(), Error>{
    // Inizializza il tracing per i log
    tracing_subscriber::fmt::init();

    let api = Api::new()?;
    let repo_id = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let weights_path = repo_id.get("model.safetensors")?;

    let tokenizer_path = repo_id.get("tokenizer.json")?;
    let config_path = repo_id.get("config.json")?;

    println!("Weights downloaded from: {:?}", weights_path);

    let device = Device::cuda_if_available(0)?;


    let cfg = serde_json::from_reader::<File, config::ModelConfig>(
        File::open(&config_path)?
    )?;

    println!("Model config: {:#?}", cfg);

    let weights = load_weights_mmap(weights_path.as_path())?;

    let tokenizer = load_tokenizer(tokenizer_path.as_path())?;

    let model = llm::models::LlamaModel::new(&weights, &cfg, &device)?;

    println!("Model loaded successfully!");

    // Create shared application state
    let state = Arc::new(AppState {
        model: Box::new(model),
        tokenizer,
        device,
        model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        eos_token_id: cfg.eos_token_id as i64,
    });

    // Create router
    let app = api::server::create_router(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("🚀 Listening http://127.0.0.1:3000");
    println!("   - Health: http://127.0.0.1:3000/health");
    println!("   - Inference: http://127.0.0.1:3000/inference");

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(())
}

fn load_weights_mmap(path: &Path) -> Result<MmapedSafetensors, Error> {
    let weights = unsafe { MmapedSafetensors::new(path)? };
    Ok(weights)
}

#[allow(unused)]
fn load_tokenizer(path: &Path) -> Result<Tokenizer, Error> {
    let tokenizer = Tokenizer::from_file(path.to_str().unwrap()).map_err(|e| anyhow::anyhow!(e))?;
    Ok(tokenizer)
}
