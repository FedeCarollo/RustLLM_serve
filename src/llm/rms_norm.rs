use candle_core::{Device, safetensors::MmapedSafetensors, Result as CandleResult, Tensor};

use crate::llm::layer::Layer;

pub struct RMSNormLayer {
    weights: Tensor,
    eps: f64,
    device: Device,
}

impl RMSNormLayer {
    pub fn new(
        weights_map: &MmapedSafetensors,
        prefix: &str,
        device: &Device,
        eps: f64,
    ) -> CandleResult<Self> {
        let weights = weights_map.load(&format!("{}.weight", prefix), device)?
            .to_dtype(candle_core::DType::F16)?;

        Ok(Self {
            weights,
            eps,
            device: device.clone(),
        })
    }
}

impl Layer for RMSNormLayer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let input = input.to_device(&self.device)?;
        let variance = input.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = variance.affine(1.0, self.eps as f64)?.sqrt()?;

        let inv_rms = rms.recip()?;
        let norm_x = input.broadcast_mul(&inv_rms)?;

        let out = norm_x.broadcast_mul(&self.weights)?;

        Ok(out)
    }
}
