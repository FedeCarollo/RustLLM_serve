pub trait Layer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor>;
}