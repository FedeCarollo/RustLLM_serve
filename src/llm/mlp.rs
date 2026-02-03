use crate::llm::layer::Layer;
use crate::llm::linear::LinearLayer;
use crate::llm::activation::Activation;
use candle_core::{Device, Tensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::Result as CandleResult;

pub struct MlpLayer {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
    activation: Activation,
    device: Device,
}

impl MlpLayer {
    pub fn new(
        weights_map: &MmapedSafetensors,
        prefix: &str,
        device: Device,
        activation: Activation,
    ) -> CandleResult<Self> {
        let gate_proj = LinearLayer::new(
            weights_map,
            &format!("{}.gate_proj", prefix),
            device.clone(),
        )?;
        let up_proj = LinearLayer::new(
            weights_map,
            &format!("{}.up_proj", prefix),
            device.clone(),
        )?;
        let down_proj = LinearLayer::new(
            weights_map,
            &format!("{}.down_proj", prefix),
            device.clone(),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
            device,
        })
    }
}

impl Layer for MlpLayer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let input = input.to_device(&self.device)?;
        let gate_output = self.gate_proj.forward(&input)?;
        let up_output = self.up_proj.forward(&input)?;
        let activated = self.activation.apply(&gate_output)?;
        let multiplied = activated.mul(&up_output)?;
        let output = self.down_proj.forward(&multiplied)?;
        Ok(output)
    }
}