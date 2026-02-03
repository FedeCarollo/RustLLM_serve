use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use std::sync::Arc;

use crate::api::models::{HealthResponse, InferenceRequest, InferenceResponse};
use crate::{llm};
use crate::app_state::AppState;

pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
    })
}

pub async fn inference(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, (StatusCode, String)> {
    tracing::info!("Inference request: {:?}", payload);

    let prompt = payload.prompt;
    let max_tokens = payload.max_tokens.min(1024); // Limita il numero massimo di token generati
    let temperature = payload.temperature.unwrap_or(1.0);
    let seed = payload.seed.unwrap_or(42);

    // Generate text
    let generated_ids = llm::inference::generate(
        &prompt,
        max_tokens,
        &state.tokenizer,
        &*state.model,
        &state.device,
        temperature,
        seed,
        state.eos_token_id,
    )
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Decode generated ids to text
    let generated_ids_u32: Vec<u32> = generated_ids.iter().map(|&id| id as u32).collect();
    let generated_text = state
        .tokenizer
        .decode(&generated_ids_u32, true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    tracing::info!("Generated text: {}", generated_text);

    Ok(Json(InferenceResponse { generated_text }))
}
