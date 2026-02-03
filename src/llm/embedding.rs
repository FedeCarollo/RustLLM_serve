use candle_core::{Device, safetensors::MmapedSafetensors, Result as CandleResult, Tensor};

use crate::llm::layer::Layer;

pub struct EmbeddingLayer {
    weights: Tensor,
    device: Device,
}

impl EmbeddingLayer {
    pub fn new(
        weights_map: &MmapedSafetensors,
        device: &Device,
        prefix: &str,
    ) -> CandleResult<Self> {
        let weights = weights_map.load(&format!("{}.weight", prefix), device)?
            .to_dtype(candle_core::DType::F16)?;

        Ok(Self {
            weights,
            device: device.clone(),
        })
    }
}

impl Layer for EmbeddingLayer {
    fn forward(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let input_ids = input_ids.to_device(&self.device)?;

        let (batch_size, seq_length) = input_ids.dims2()?;

        let flat_input = input_ids.reshape(&[batch_size * seq_length])?;

        let flat_embeddings = self.weights.index_select(&flat_input, 0)?;

        let embeddings = flat_embeddings.reshape(&[batch_size, seq_length, self.weights.dim(1)?])?;

        Ok(embeddings)
    }
}