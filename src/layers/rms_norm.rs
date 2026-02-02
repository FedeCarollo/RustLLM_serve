use candle_core::{Device, safetensors::MmapedSafetensors, Result as CandleResult, Tensor};

use crate::layers::layer::Layer;

pub struct RMSNormLayer {
    weights: Tensor,
    eps: f64,
    prefix: String,
    device: Device,
}

impl RMSNormLayer {
    pub fn new(
        weights_map: &MmapedSafetensors,
        prefix: &str,
        device: &Device,
        eps: f64,
    ) -> CandleResult<Self> {
        let weights = weights_map.load(&format!("{}.weight", prefix), device)?;

        Ok(Self {
            weights,
            eps,
            prefix: String::from(prefix),
            device: device.clone(),
        })
    }
}

impl Layer for RMSNormLayer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let orig_dtype = input.dtype();

        let input = input.to_dtype(candle_core::DType::F32)?;
        let variance = input.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = variance.affine(1.0, self.eps as f64)?.sqrt()?;

        let inv_rms = rms.recip()?;
        let norm_x = input.broadcast_mul(&inv_rms)?;

        let norm_x_orig = norm_x.to_dtype(orig_dtype)?;

        let out = norm_x_orig.broadcast_mul(&self.weights)?;

        Ok(out)
    }
}
