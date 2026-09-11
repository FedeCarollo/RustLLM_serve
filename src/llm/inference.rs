use candle_core::{Device, IndexOp};
use candle_nn::ops::{softmax};
use rand::{SeedableRng, distr::{Distribution, weighted::WeightedIndex}};
use tokenizers::Tokenizer;
use candle_core::Result as CandleResult;

use crate::llm;

pub fn generate(
    prompt: &str, 
    max_tokens: usize, 
    tokenizer: &Tokenizer, 
    model: &dyn llm::models::Model, 
    device: &Device, 
    temperature: f64, 
    seed: u64,
    eos_token_id: i64
) -> CandleResult<Vec<i64>> {

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let temperature = temperature.max(0.01).min(100.0); // Clamp temperature to a reasonable range

    let encoding = tokenizer.encode(prompt, true).map_err(|e| candle_core::Error::msg(e.to_string()))?;
    let mut input_ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<i64>>();

    // Initialize state
    model.clear_cache();

    // First forward pass with all prompt tokens
    let input_tensor = candle_core::Tensor::from_slice(&input_ids, &[1, input_ids.len()], device)?;
    let output = model.forward_with_cache(&input_tensor)?;
    
    // Get last token logits
    let mut next_token_logits = output.i((0, output.dim(1)? - 1))?;

    // Apply temperature scaling
    let mut scaled_logits = (next_token_logits / temperature)?;
    
    // Sample next token
    let mut prob_dist = softmax(&scaled_logits, 0)?;
    let mut dist = WeightedIndex::new(prob_dist.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?).map_err(|e| candle_core::Error::msg(e.to_string()))?;
    let mut next_token = dist.sample(&mut rng) as i64;

    if next_token == eos_token_id {
        return Ok(input_ids);
    }
    
    input_ids.push(next_token);

    // Generate remaining tokens one at a time using KV cache
    for _ in 1..max_tokens {
        // Only pass the last token through the model
        let input_tensor = candle_core::Tensor::from_slice(&[next_token], &[1, 1], device)?;
        let output = model.forward_with_cache(&input_tensor)?;
        
        // Get last token logits
        next_token_logits = output.i((0, output.dim(1)? - 1))?;

        // Apply temperature scaling
        scaled_logits = (next_token_logits / temperature)?;
        
        // Sample next token
        prob_dist = softmax(&scaled_logits, 0)?;
        dist = WeightedIndex::new(prob_dist.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        next_token = dist.sample(&mut rng) as i64;

        if next_token == eos_token_id {
            break;
        }
        
        input_ids.push(next_token);
    }

    Ok(input_ids)
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use hf_hub::api::sync::Api;
    use std::fs::File;
    use crate::config::ModelConfig;
    use crate::llm::models::{LlamaModel, Model};
    

    #[test]
    fn test_generation_and_cache_length() -> anyhow::Result<()> {
        let api = Api::new()?;
        let repo_id = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());
        
        let weights_path = repo_id.get("model.safetensors")?;
        let tokenizer_path = repo_id.get("tokenizer.json")?;
        let config_path = repo_id.get("config.json")?;
        
        let device = Device::Cpu; // Run test on CPU to avoid CUDA dependency in tests
        let cfg = serde_json::from_reader::<File, ModelConfig>(File::open(&config_path)?)?;
        let weights = unsafe { candle_core::safetensors::MmapedSafetensors::new(weights_path.as_path())? };
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path.to_str().unwrap()).unwrap();
        
        let model = LlamaModel::new(&weights, &cfg, &device, candle_core::DType::F32)?;
        
        let prompt = "Test prompt";
        let max_tokens = 5;
        
        let generated_ids = generate(
            prompt,
            max_tokens,
            &tokenizer,
            &model,
            &device,
            1.0,
            42,
            cfg.eos_token_id as i64,
        ).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        
        // Assert that we generated exactly max_tokens (plus the prompt tokens, or maybe generate returns just the full sequence)
        // Assert that we generated tokens
        assert!(generated_ids.len() > 2);
        
        // Assert that cache length equals generated sequence length
        let cache_len = model.get_cache_len();
        assert_eq!(cache_len, generated_ids.len());
        
        Ok(())
    }
}
