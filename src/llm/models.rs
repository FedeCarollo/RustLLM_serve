use crate::config::ModelConfig;
use crate::llm;
use crate::llm::decoder::DecoderLayer;
use crate::llm::rms_norm::RMSNormLayer;
use crate::llm::lm_head::LMHeadLayer;
use crate::llm::layer::Layer;

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_core::safetensors::MmapedSafetensors;

use llm::embedding::EmbeddingLayer;


#[allow(dead_code)]
pub trait Model {
    fn load(weights: &MmapedSafetensors, config: &ModelConfig, device: &Device, dtype: candle_core::DType) -> CandleResult<Self> where Self: Sized;
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor>;
    fn forward_with_cache(&self, input_ids: &Tensor) -> CandleResult<Tensor>;
    fn clear_cache(&self);
    fn get_cache_len(&self) -> usize;
    fn num_layers(&self) -> usize;
}

#[allow(unused)]
pub struct LlamaModel {
    pub device: Device,
    pub embed_layer: EmbeddingLayer,
    pub decoder_layers: Vec<DecoderLayer>,
    pub norm_layer: RMSNormLayer,
    pub lm_head: LMHeadLayer,
    pub cfg: ModelConfig,
}

impl LlamaModel {
    pub fn new(
        weights: &MmapedSafetensors,
        config: &ModelConfig,
        device: &Device,
        dtype: candle_core::DType,
    ) -> CandleResult<Self> {
        let embed_layer = EmbeddingLayer::new(weights, device, "model.embed_tokens", dtype)?;
        
        let max_seq_len = config.max_position_embeddings;

        let layers = (0..config.num_hidden_layers).map(|layer_idx| {
            DecoderLayer::new(
                weights,
                &format!("model.layers.{}", layer_idx),
                config,
                device,
                dtype,
                max_seq_len,
            )
        }).collect::<CandleResult<Vec<_>>>()?;

        let norm_layer = RMSNormLayer::new(
            weights,
            "model.norm",
            device,
            config.rms_norm_eps as f64,
            dtype,
        )?;

        let lm_head = LMHeadLayer::new(
            weights,
            "lm_head",
            device,
            dtype,
        )?;

        Ok(Self {
            device: device.clone(),
            embed_layer,
            decoder_layers: layers,
            norm_layer,
            lm_head,
            cfg: config.clone(),
        })
    }
}

#[allow(unused)]
impl Model for LlamaModel {
    fn load(weights: &MmapedSafetensors, config: &ModelConfig, device: &Device, dtype: candle_core::DType) -> CandleResult<Self> {
        Self::new(weights, config, device, dtype)
    }
    
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let input_ids = input_ids.to_device(&self.device)?;
        let embeddings = self.embed_layer.forward(&input_ids)?;

        let mut hidden_states = embeddings;
        for layer in self.decoder_layers.iter() {
            hidden_states = layer.forward(&hidden_states)?;
        }

        let normed_output = self.norm_layer.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&normed_output)?;

        Ok(logits)
    }

    fn forward_with_cache(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let input_ids = input_ids.to_device(&self.device)?;
        let embeddings = self.embed_layer.forward(&input_ids)?;

        let mut hidden_states = embeddings;
        for layer in self.decoder_layers.iter() {
            hidden_states = layer.forward_with_cache(&hidden_states)?;
        }

        let normed_output = self.norm_layer.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&normed_output)?;

        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.decoder_layers.len()
    }

    fn clear_cache(&self) {
        for layer in self.decoder_layers.iter() {
            layer.self_attn.clear_cache();
        }
    }

    fn get_cache_len(&self) -> usize {
        self.decoder_layers.first().map(|l| l.self_attn.get_cache_len()).unwrap_or(0)
    }
}