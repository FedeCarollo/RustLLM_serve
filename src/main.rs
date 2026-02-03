use std::fs::File;

use anyhow::Error;
use candle_core::{Device, IndexOp, Tensor, safetensors::MmapedSafetensors};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use std::path::Path;

use crate::layers::layer::Layer;
mod config;
mod models;
mod layers;

#[allow(unused)]
fn main() -> Result<(), Error>{
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

    let model = models::LlamaModel::new(&weights, &cfg, &device)?;

    let prompt = "Hello, my name is";
    let output_ids = generate(prompt, &tokenizer, &model, &device, 10, cfg.eos_token_id as i64)?;
    println!("Input prompt: {}", prompt);
    println!("Output IDs: {:?}", output_ids);
    
    // Decode output IDs to text
    let output_ids_u32: Vec<u32> = output_ids.iter().map(|&id| id as u32).collect();
    let output_text = tokenizer.decode(&output_ids_u32, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    println!("Generated text: {}", output_text);


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

#[allow(dead_code)]
fn inference(text: &str, tokenizer: &Tokenizer, model: &impl models::Model, device: &Device) -> Result<Vec<i64>, Error> {
    let encoding = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<i64>>();

    let input_tensor = Tensor::from_slice(&input_ids, &[1, input_ids.len()], device)?;

    let output = model.forward(&input_tensor)?;

    let output_ids = output
        .argmax(candle_core::D::Minus1)?
        .to_dtype(candle_core::DType::I64)?
        .squeeze(0)?
        .to_vec1()?;

    Ok(output_ids)
}

#[allow(dead_code)]
fn generate(
    text: &str, 
    tokenizer: &Tokenizer, 
    model: &impl models::Model, 
    device: &Device,
    max_new_tokens: usize,
    eos_token_id: i64,
) -> Result<Vec<i64>, Error> {
    let encoding = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
    let mut input_ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<i64>>();

    for _ in 0..max_new_tokens {
        let input_tensor = Tensor::from_slice(&input_ids, &[1, input_ids.len()], device)?;
        let output = model.forward(&input_tensor)?;
        
        // Prendi solo l'ultimo token (ultima posizione della sequenza)
        let next_token_logits = output.i((0, output.dim(1)? - 1))?;
        let next_token = next_token_logits.argmax(0)?.to_dtype(candle_core::DType::I64)?.to_vec0::<i64>()?;

        if next_token == eos_token_id {
            break;
        }
        
        input_ids.push(next_token);
    }

    Ok(input_ids)
}

#[allow(dead_code)]
fn test_self_attention(
    text: &str, 
    tokenizer: &Tokenizer, 
    model: &models::LlamaModel, 
    device: &Device
) -> Result<(), Error> {
    println!("\n=== Testing Self Attention ===");
    
    // Tokenize input
    let encoding = tokenizer.encode(text, true).map_err(|e| anyhow::anyhow!(e))?;
    let input_ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<i64>>();
    let input_tensor = Tensor::from_slice(&input_ids, &[1, input_ids.len()], device)?;
    
    println!("Input IDs shape: {:?}", input_tensor.shape());
    println!("Input IDs: {:?}", input_ids);
    
    // Get embeddings
    let embeddings = model.embed_layer.forward(&input_tensor)?;
    println!("Embeddings shape: {:?}", embeddings.shape());
    
    // Test first decoder layer's self attention
    let first_layer = &model.decoder_layers[0];
    
    // Apply input norm
    let normed_input = first_layer.input_norm.forward(&embeddings)?;
    println!("Normed input shape: {:?}", normed_input.shape());
    
    // Test self attention
    println!("\nTesting self attention forward...");
    let attn_output = first_layer.self_attn.forward(&normed_input)?;
    println!("Attention output shape: {:?}", attn_output.shape());
    
    // Print some values
    let attn_flat = attn_output.flatten_all()?.to_vec1::<f64>()?;
    println!("First 10 attention output values: {:?}", &attn_flat[..10.min(attn_flat.len())]);
    
    println!("\n=== Self Attention Test Complete ===");
    Ok(())
}