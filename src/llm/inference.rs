use candle_core::{Device, IndexOp};
use candle_nn::ops::{softmax};
use rand::{SeedableRng, distr::{Distribution, weighted::WeightedIndex}};
use tokenizers::Tokenizer;
use candle_core::Result as CandleResult;

use crate::llm;
use crate::llm::causal_self_attention::KVCache;

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

    // Initialize KV caches for all layers (number of layers should match the model)
    let num_layers = model.num_layers();
    let mut kv_caches: Vec<KVCache> = vec![KVCache::new(); num_layers];

    // First forward pass with all prompt tokens
    let input_tensor = candle_core::Tensor::from_slice(&input_ids, &[1, input_ids.len()], device)?;
    let output = model.forward_with_cache(&input_tensor, &mut kv_caches, 0)?;
    
    // Get last token logits
    let mut next_token_logits = output.i((0, output.dim(1)? - 1))?;

    // Apply temperature scaling
    let mut scaled_logits = (next_token_logits / temperature)?;
    
    // Sample next token
    let mut prob_dist = softmax(&scaled_logits, 0)?;
    let mut dist = WeightedIndex::new(prob_dist.to_vec1::<f64>()?).map_err(|e| candle_core::Error::msg(e.to_string()))?;
    let mut next_token = dist.sample(&mut rng) as i64;

    if next_token == eos_token_id {
        return Ok(input_ids);
    }
    
    input_ids.push(next_token);
    let mut position = input_ids.len() - 1;

    // Generate remaining tokens one at a time using KV cache
    for _ in 1..max_tokens {
        // Only pass the last token through the model
        let input_tensor = candle_core::Tensor::from_slice(&[next_token], &[1, 1], device)?;
        let output = model.forward_with_cache(&input_tensor, &mut kv_caches, position)?;
        
        // Get last token logits
        next_token_logits = output.i((0, output.dim(1)? - 1))?;

        // Apply temperature scaling
        scaled_logits = (next_token_logits / temperature)?;
        
        // Sample next token
        prob_dist = softmax(&scaled_logits, 0)?;
        dist = WeightedIndex::new(prob_dist.to_vec1::<f64>()?).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        next_token = dist.sample(&mut rng) as i64;

        if next_token == eos_token_id {
            break;
        }
        
        input_ids.push(next_token);
        position += 1;
    }

    Ok(input_ids)
}