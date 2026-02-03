use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub seed: Option<u64>,
}

fn default_max_tokens() -> usize {
    50
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub generated_text: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
}
