use candle_core::{Device, Result as CandleResult};
use candle_core::safetensors::MmapedSafetensors;
use crate::llm::activation::Activation;
use crate::llm::layer::Layer;
use crate::{config, llm::{causal_self_attention::{CausalSelfAttentionLayer, KVCache}, mlp::MlpLayer, rms_norm::RMSNormLayer}};


pub struct DecoderLayer {
    pub self_attn: CausalSelfAttentionLayer,
    pub mlp: MlpLayer,
    pub input_norm: RMSNormLayer,
    pub post_attention_norm: RMSNormLayer,
    pub device: Device,
}

impl DecoderLayer {
    pub fn new(
        weights: &MmapedSafetensors,
        prefix: &str,
        config: &config::ModelConfig,
        device: &Device
    ) -> CandleResult<Self> {
        let self_attn = CausalSelfAttentionLayer::new(
            weights,
            &format!("{}.self_attn", prefix),
            config.num_attention_heads,
            config.num_key_value_heads,
            config.hidden_size,
            config.rope_theta,
            device.clone(),
        )?;

        let mlp = MlpLayer::new(
            weights,
            &format!("{}.mlp", prefix),
            device.clone(),
            Activation::from(&config.hidden_activation),
        )?;

        let input_norm = RMSNormLayer::new(
            weights,
            &format!("{}.input_layernorm", prefix),
            device,
            config.rms_norm_eps as f64,
        )?;

        let post_attention_norm = RMSNormLayer::new(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
            device,
            config.rms_norm_eps as f64,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_norm,
            post_attention_norm,
            device: device.clone(),
        })
        
    }

    pub fn forward_with_cache(
        &self,
        input: &candle_core::Tensor,
        kv_cache: &mut KVCache,
        position: usize,
    ) -> CandleResult<candle_core::Tensor> {
        let input = input.to_device(&self.device)?;
        let normed_input = self.input_norm.forward(&input)?;
        let attn_output = self.self_attn.forward_with_cache(&normed_input, kv_cache, position)?;
        let attn_residual = input.add(&attn_output)?;

        let normed_attn = self.post_attention_norm.forward(&attn_residual)?;
        let mlp_output = self.mlp.forward(&normed_attn)?;
        let output = attn_residual.add(&mlp_output)?;

        Ok(output)
    }
}

impl Layer for DecoderLayer {
    fn forward(&self, input: &candle_core::Tensor) -> CandleResult<candle_core::Tensor> {
        let input = input.to_device(&self.device)?;
        println!("Input moved to device.");
        let normed_input = self.input_norm.forward(&input)?;
        println!("Input normalized.");
        let attn_output = self.self_attn.forward(&normed_input)?;
        println!("Self-attention output computed.");
        let attn_residual = input.add(&attn_output)?;
        println!("Attention residual added.");

        let normed_attn = self.post_attention_norm.forward(&attn_residual)?;
        println!("Post-attention normalization completed.");
        let mlp_output = self.mlp.forward(&normed_attn)?;
        println!("MLP output computed.");
        let output = attn_residual.add(&mlp_output)?;
        println!("Final output computed.");

        Ok(output)
    }
}