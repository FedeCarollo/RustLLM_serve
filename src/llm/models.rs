use crate::config::ModelConfig;
use crate::llm;
use crate::llm::decoder::DecoderLayer;
use crate::llm::rms_norm::RMSNormLayer;
use crate::llm::lm_head::LMHeadLayer;
use crate::llm::layer::Layer;

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_core::safetensors::MmapedSafetensors;

use llm::embedding::EmbeddingLayer;


#[allow(unused)]
pub trait Model {
    fn load(weights: &MmapedSafetensors, config: &ModelConfig, device: &Device) -> CandleResult<Self> where Self: Sized;
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor>;
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
        let input_ids = input_ids.to_device(&self.device)?;
        let embeddings = self.embed_layer.forward(&input_ids)?;

        // let embeddings = embeddings.to_dtype(candle_core::DType::F16)?;

        let mut hidden_states = embeddings;
        for layer in &self.decoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        println!("Hidden states completed all decoder layers.");

        let normed_output = self.norm_layer.forward(&hidden_states)?;

        println!("Output normalized.");

        let logits = self.lm_head.forward(&normed_output)?;

        println!("Logits computed.");

        Ok(logits)
    }
}