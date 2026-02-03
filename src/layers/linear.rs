use candle_core::{Device, Tensor, Result as CandleResult};
use candle_core::safetensors::MmapedSafetensors;
use candle_nn::var_builder::SimpleBackend;
use crate::layers::layer::Layer;


pub struct LinearLayer {
    weights: Tensor,
    bias: Option<Tensor>,
    device: Device,
}

impl LinearLayer {
    pub fn new(
        weights_map: &MmapedSafetensors, 
        prefix: &str, 
        device: Device,
    ) -> CandleResult<Self> {
        let weights = weights_map.load(&format!("{}.weight", prefix), &device)?
            .to_dtype(candle_core::DType::F16)?;

        let bias = if weights_map.contains_tensor(&format!("{}.bias", prefix)) {
            Some(weights_map.load(&format!("{}.bias", prefix), &device)?)
        } else {
            None
        };

        Ok(Self {
            weights,
            bias,
            device,
        })
    }
}

impl Layer for LinearLayer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let input = input.to_device(&self.device)?;
        let mut output = input.broadcast_matmul(&self.weights.t()?)?;
        if let Some(bias) = &self.bias {
            output = output.broadcast_add(bias)?
        }
        Ok(output)
    }
}