use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub enum Activation {
    Gelu,
    Relu,
    Silu
}

impl Activation {
    pub fn from_str(name: &str) -> Self {
        match name {
            "gelu" => Activation::Gelu,
            "relu" => Activation::Relu,
            "silu" => Activation::Silu,
            _ => panic!("Unsupported activation function: {}", name),
        }
    }

    pub fn apply(&self, input: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        match self {
            Activation::Gelu => input.gelu(),
            Activation::Relu => input.relu(),
            Activation::Silu => input.silu(),
        }
    }
}