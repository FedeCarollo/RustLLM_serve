use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct ModelConfig {
    architectures: Vec<String>,
    attention_bias: bool,
    bos_token_id: i32,
    eos_token_id: i32,
    
    #[serde(rename = "hidden_act")]
    hidden_activation: String,

    hidden_size: usize,
    initializer_range: f32,

    intermediate_size: usize,
    max_position_embeddings: usize,

    model_type: String,

    num_attention_heads: usize,
    num_hidden_layers: usize,

    num_key_value_heads: usize,
    pretraining_tp: usize,

    rms_norm_eps: f32,

    // rope_scaling: Option<>,

    rope_theta: f32,

    tie_word_embeddings: bool,

    torch_dtype: String,

    transformers_version: String,

    use_cache: bool,
    vocab_size: usize,
}
