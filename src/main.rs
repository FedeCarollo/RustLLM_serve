use std::fs::File;

use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use hf_hub::api::sync::Api;
mod config;

fn main() -> Result<(), Error>{
    let api = Api::new()?;
    let repo_id = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let weights_path = repo_id.get("model.safetensors")?;
    let tokenizer_path = repo_id.get("tokenizer.json")?;
    let config_path = repo_id.get("config.json")?;

    println!("Weights downloaded from: {:?}", weights_path);


    let cfg = serde_json::from_reader::<File, config::ModelConfig>(
        File::open(&config_path)?
    )?;

    println!("Model config: {:#?}", cfg);


    Ok(())
}
