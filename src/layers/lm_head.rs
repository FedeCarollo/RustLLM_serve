use candle_core::{Device, Tensor, Result as CandleResult};
use candle_core::safetensors::MmapedSafetensors;

use crate::layers::layer::Layer;

pub struct LMHeadLayer {
    weights: Tensor,
    device: Device
}

impl LMHeadLayer {
    pub fn new(weights: &MmapedSafetensors, prefix: &str, device: &Device) -> CandleResult<Self> {
        let weights = weights.load(&format!("{}.weight", prefix), device)?
            .to_dtype(candle_core::DType::F16)?;
        Ok(Self {
            weights,
            device: device.clone(),            
        })
    }
}

impl Layer for LMHeadLayer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let input = input.to_device(&self.device)?;
        let output = input.broadcast_matmul(&self.weights.t()?)?;
        Ok(output)
    }
}

