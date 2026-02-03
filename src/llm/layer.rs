use candle_core::Tensor;
use candle_core::Result as CandleResult;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor>;
}
