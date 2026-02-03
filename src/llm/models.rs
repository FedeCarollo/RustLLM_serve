use crate::config::ModelConfig;
use crate::llm;
use crate::llm::decoder::DecoderLayer;
use crate::llm::rms_norm::RMSNormLayer;
use crate::llm::lm_head::LMHeadLayer;
use crate::llm::layer::Layer;
use crate::llm::causal_self_attention::KVCache;

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_core::safetensors::MmapedSafetensors;

use llm::embedding::EmbeddingLayer;


#[allow(unused)]
pub trait Model {
    fn load(weights: &MmapedSafetensors, config: &ModelConfig, device: &Device) -> CandleResult<Self> where Self: Sized;
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor>;
    fn forward_with_cache(&self, input_ids: &Tensor, kv_caches: &mut Vec<KVCache>, position: usize) -> CandleResult<Tensor>;
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
    ) -> CandleResult<Self> {
        let embed_layer = EmbeddingLayer::new(weights, device, "model.embed_tokens")?;

        let layers = (0..config.num_hidden_layers).map(|layer_idx| {
            DecoderLayer::new(
                weights,
                &format!("model.layers.{}", layer_idx),
                config,
                device,
            )
        }).collect::<CandleResult<Vec<_>>>()?;

        // let layers = (0..1).map(|layer_idx| {
        //     DecoderLayer::new(
        //         weights,
        //         &format!("model.layers.{}", layer_idx),
        //         config,
        //         device,
        //     )
        // }).collect::<CandleResult<Vec<_>>>()?;

        let norm_layer = RMSNormLayer::new(
            weights,
            "model.norm",
            device,
            config.rms_norm_eps as f64,
        )?;

        let lm_head = LMHeadLayer::new(
            weights,
            "lm_head",
            device,
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
    fn load(weights: &MmapedSafetensors, config: &ModelConfig, device: &Device) -> CandleResult<Self> {
        Self::new(weights, config, device)
    }
    
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let mut kv_caches = vec![KVCache::new(); self.decoder_layers.len()];
        self.forward_with_cache(input_ids, &mut kv_caches, 0)
    }

    fn forward_with_cache(&self, input_ids: &Tensor, kv_caches: &mut Vec<KVCache>, position: usize) -> CandleResult<Tensor> {
        let input_ids = input_ids.to_device(&self.device)?;
        let embeddings = self.embed_layer.forward(&input_ids)?;

        let mut hidden_states = embeddings;
        for (idx, layer) in self.decoder_layers.iter().enumerate() {
            hidden_states = layer.forward_with_cache(&hidden_states, &mut kv_caches[idx], position)?;
        }

        let normed_output = self.norm_layer.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&normed_output)?;

        Ok(logits)
    }

    fn num_layers(&self) -> usize {
        self.decoder_layers.len()
    }
}