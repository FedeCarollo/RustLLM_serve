use candle_core::Device;
use tokenizers::Tokenizer;

use crate::llm;

pub struct AppState {
    pub model: Box<dyn llm::models::Model + Send + Sync>,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub model_name: String,
    pub eos_token_id: i64,
}