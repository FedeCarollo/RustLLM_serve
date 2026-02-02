use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
#[allow(unused)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    #[serde(rename = "hidden_act")]
    pub hidden_activation: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f32,
    // rope_scaling: Option<>,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}
