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

    for _ in 0..max_tokens {
        let input_tensor = candle_core::Tensor::from_slice(&input_ids, &[1, input_ids.len()], device)?;
        let output = model.forward(&input_tensor)?;
        
        // Get last token logits
        let next_token_logits = output.i((0, output.dim(1)? - 1))?;

        // Apply temperature scaling
        let scaled_logits = (next_token_logits / temperature)?;
        
        // Sample next token according to the probabilities
        let prob_dist = softmax(&scaled_logits, 0)?;

        let dist = WeightedIndex::new(prob_dist.to_vec1::<f64>()?).map_err(|e| candle_core::Error::msg(e.to_string()))?;

        let next_token = dist.sample(&mut rng) as i64;

        if next_token == eos_token_id {
            break;
        }
        
        input_ids.push(next_token);
    }

    Ok(input_ids)
}