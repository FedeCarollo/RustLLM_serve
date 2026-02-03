use std::fs::File;
use std::time::Instant;

use anyhow::Error;
use candle_core::IndexOp;
use candle_core::{Device, safetensors::MmapedSafetensors};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use std::path::Path;

use RustLLM_serve::llm::models::{Model, LlamaModel};
use RustLLM_serve::llm::causal_self_attention::KVCache;
use RustLLM_serve::config;

fn main() -> Result<(), Error> {
    println!("🔬 KV Cache Benchmark\n");
    
    let api = Api::new()?;
    let repo_id = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

    let weights_path = repo_id.get("model.safetensors")?;
    let tokenizer_path = repo_id.get("tokenizer.json")?;
    let config_path = repo_id.get("config.json")?;

    println!("📦 Loading model from HuggingFace...");
    
    let device = Device::cuda_if_available(0)?;
    println!("🖥️  Device: {:?}", device);

    let cfg = serde_json::from_reader::<File, config::ModelConfig>(
        File::open(&config_path)?
    )?;

    let weights = load_weights_mmap(weights_path.as_path())?;
    let tokenizer = load_tokenizer(tokenizer_path.as_path())?;
    let model = LlamaModel::new(&weights, &cfg, &device)?;

    println!("✅ Model loaded successfully!\n");

    // ========================================
    // BENCHMARK: Forward vs Forward with Cache
    // ========================================
    
    let prompt = "Hello, my name is";
    println!("🔬 Starting benchmark with prompt: \"{}\"", prompt);
    
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    
    println!("📝 Prompt tokens: {} tokens", input_ids.len());
    println!("\n--- Testing generation of 50 tokens ---\n");

    // ========================================
    // TEST 1: Forward WITHOUT cache (baseline)
    // ========================================
    println!("📊 Test 1: Forward WITHOUT KV cache");
    println!("   (Full attention computation every time)");
    
    let mut test_ids = input_ids.clone();
    let start_no_cache = Instant::now();
    
    for i in 0..1 {
        let test_tensor = candle_core::Tensor::from_slice(&test_ids, &[1, test_ids.len()], &device)?;
        let output = model.forward(&test_tensor)?;
        
        // Get last token (greedy decoding for benchmark)
        let logits = output.i((0, output.dim(1)? - 1))?;
        let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
        test_ids.push(next_token);
        
        if (i + 1) % 10 == 0 {
            println!("   Generated {} tokens...", i + 1);
        }
    }
    
    let time_no_cache = start_no_cache.elapsed();
    println!("   ✅ Completed in: {:.2?}", time_no_cache);
    println!("   Average time per token: {:.2?}", time_no_cache / 50);

    // ========================================
    // TEST 2: Forward WITH cache
    // ========================================
    println!("\n📊 Test 2: Forward WITH KV cache");
    println!("   (Caching key/value for efficient generation)");
    
    let mut kv_caches: Vec<KVCache> = vec![KVCache::new(); model.num_layers()];
    let mut cached_ids = input_ids.clone();
    
    let start_with_cache = Instant::now();
    
    // First pass: process all prompt tokens
    let prompt_tensor = candle_core::Tensor::from_slice(&cached_ids, &[1, cached_ids.len()], &device)?;
    let mut output = model.forward_with_cache(&prompt_tensor, &mut kv_caches, 0)?;
    
    // Get first generated token
    let mut logits = output.i((0, output.dim(1)? - 1))?;
    let mut next_token = logits.argmax(0)?.to_scalar::<u32>()?;
    cached_ids.push(next_token);
    
    println!("   Prompt processed, generating tokens...");
    
    // Subsequent passes: only process one new token at a time
    for i in 1..2 {
        let single_token_tensor = candle_core::Tensor::from_slice(&[next_token], &[1, 1], &device)?;
        let position = cached_ids.len() - 1;
        output = model.forward_with_cache(&single_token_tensor, &mut kv_caches, position)?;
        
        logits = output.i((0, output.dim(1)? - 1))?;
        next_token = logits.argmax(0)?.to_scalar::<u32>()?;
        cached_ids.push(next_token);
        
        if (i + 1) % 10 == 0 {
            println!("   Generated {} tokens...", i + 1);
        }
    }
    
    let time_with_cache = start_with_cache.elapsed();
    println!("   ✅ Completed in: {:.2?}", time_with_cache);
    println!("   Average time per token: {:.2?}", time_with_cache / 50);

    // ========================================
    // RESULTS SUMMARY
    // ========================================
    println!("\n{}", "=".repeat(60));
    println!("📈 BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));
    println!("Without KV Cache: {:.2?} ({:.2?}/token)", time_no_cache, time_no_cache / 50);
    println!("With KV Cache:    {:.2?} ({:.2?}/token)", time_with_cache, time_with_cache / 50);
    println!("{}", "-".repeat(60));
    
    let speedup = time_no_cache.as_secs_f64() / time_with_cache.as_secs_f64();
    println!("⚡ Speedup: {:.2}x faster with KV cache! 🚀", speedup);
    
    let time_saved = time_no_cache.checked_sub(time_with_cache).unwrap_or_default();
    let percent_saved = (time_saved.as_secs_f64() / time_no_cache.as_secs_f64()) * 100.0;
    println!("💾 Time saved: {:.2?} ({:.1}%)", time_saved, percent_saved);
    println!("{}", "=".repeat(60));
    
    println!("\n✨ Benchmark completed!\n");

    Ok(())
}

fn load_weights_mmap(path: &Path) -> Result<MmapedSafetensors, Error> {
    let weights = unsafe { MmapedSafetensors::new(path)? };
    Ok(weights)
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, Error> {
    let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;
    Ok(tokenizer)
}
