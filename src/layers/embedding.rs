use candle_core::{Device, safetensors::MmapedSafetensors, Result as CandleResult, Tensor};

use crate::layers::layer::Layer;

pub struct EmbeddingLayer {
    weights: Tensor,
    name: String,
    device: Device,
}

impl EmbeddingLayer {
    pub fn new(
        weights_map: &MmapedSafetensors,
        device: &Device,
        name: &str,
    ) -> CandleResult<Self> {
        let weights = weights_map.load(name, device)?;
        Ok(Self {
            weights,
            name: String::from(name),
            device: device.clone(),
        })
    }
}

impl Layer for EmbeddingLayer {
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let embeddings = self.weights.index_select(input_ids, 0)?.to_device(&self.device)?;
        Ok(embeddings)
    }
}