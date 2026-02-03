use crate::llm::layer::Layer;
use crate::llm::linear::LinearLayer;
use candle_core::{Device, Tensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::Result as CandleResult;

/// KV cache for storing key and value tensors across generation steps
#[derive(Clone)]
pub struct KVCache {
    pub k_cache: Option<Tensor>,
    pub v_cache: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
        }
    }

    #[allow(unused)]
    pub fn clear(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

pub struct CausalSelfAttentionLayer {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    n_heads: usize,
    n_kv_heads: usize,
    hidden_size: usize,
    rope_theta: f32,
    device: Device,
}

impl CausalSelfAttentionLayer {
    pub fn new(
        weights: &MmapedSafetensors,
        prefix: &str,
        n_heads: usize,
        n_kv_heads: usize,
        hidden_size: usize,
        rope_theta: f32,
        device: Device,
    ) -> CandleResult<Self> {
        let q_proj = LinearLayer::new(weights, &format!("{}.q_proj", prefix), device.clone())?;
        let k_proj = LinearLayer::new(weights, &format!("{}.k_proj", prefix), device.clone())?;
        let v_proj = LinearLayer::new(weights, &format!("{}.v_proj", prefix), device.clone())?;
        let o_proj = LinearLayer::new(weights, &format!("{}.o_proj", prefix), device.clone())?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            hidden_size,
            rope_theta,
            device,
        })
    }

    fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
        let last_dim = x.dim(candle_core::D::Minus1)?; // Get the size of the last dimension

        let xs1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?; // First half
        let xs2 = x.narrow(
            candle_core::D::Minus1,
            last_dim / 2,
            last_dim - last_dim / 2,
        )?; // Second half

        let neg_xs2 = xs2.neg()?; // Negate the second half
        Tensor::cat(&[neg_xs2, xs1], candle_core::D::Minus1) // Concatenate along the last dimension
    }

    fn apply_rotary_emb(
        q: &Tensor,
        k: &Tensor,
        seq_len: usize,
        head_dim: usize,
        rope_theta: f32,
        device: &Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let half_head_dim = head_dim / 2;

        let exponents = (0..half_head_dim)
            .map(|i| 2.0 * (i as f64) / (head_dim as f64))
            .collect::<Vec<f64>>();

        let inv_freq: Vec<f32> = exponents
            .iter()
            .map(|&exp| 1.0f32 / (rope_theta.powf(exp as f32) as f32))
            .collect();

        let inv_freq_tensor = Tensor::new(inv_freq.as_slice(), device)?;

        // Compute angle(t)

        let t = Tensor::arange(0u32, seq_len as u32, device)?
            .to_dtype(candle_core::DType::F32)?; //(seq_len,)

        let freqs = t
            .unsqueeze(1)? // (seq_len, 1)
            .broadcast_mul(&inv_freq_tensor.unsqueeze(0)?)?; // (seq_len, half_head_dim)

        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // (seq_len, head_dim)

        let cos = emb.cos()?.to_dtype(candle_core::DType::F16)?;
        let sin = emb.sin()?.to_dtype(candle_core::DType::F16)?;

        // final formula: x' = x * cos + rotate_half(x) * sin

        let q_rotated = (q.broadcast_mul(&cos)? + Self::rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_rotated = (k.broadcast_mul(&cos)? + Self::rotate_half(k)?.broadcast_mul(&sin)?)?;

        Ok((q_rotated, k_rotated))
    }

    fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
        if n_rep == 1 {
            return Ok(x);
        }

        let (b_sz, n_kv_heads, seq_len, head_dim) = x.shape().dims4()?;
        let x = x
            .unsqueeze(2)? // (b_sz, n_kv_heads, 1, seq_len, head_dim)
            .expand(&[b_sz, n_kv_heads, n_rep, seq_len, head_dim])? // (b_sz, n_kv_heads, n_rep, seq_len, head_dim)
            .reshape(&[b_sz, n_kv_heads * n_rep, seq_len, head_dim])?; // (b_sz, n_kv_heads * n_rep, seq_len, head_dim)
        Ok(x)
    }

    fn create_causal_mask(seq_len: usize, device: &Device) -> CandleResult<Tensor> {
        // Causal mask
        let mask = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect::<Vec<f32>>();

        let mask_tensor = Tensor::from_vec(mask, (seq_len, seq_len), device)?
            .to_dtype(candle_core::DType::F16)?;
        let mask_tensor = mask_tensor.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, seq_len)

        Ok(mask_tensor)
    }
}

impl CausalSelfAttentionLayer {
    /// Forward pass with KV cache support
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
        kv_cache: &mut KVCache,
        position: usize,
    ) -> CandleResult<Tensor> {
        let input = input.to_device(&self.device)?;
        let q = self.q_proj.forward(&input)?;
        let k_new = self.k_proj.forward(&input)?;
        let v_new = self.v_proj.forward(&input)?;

        let head_dim = self.hidden_size / self.n_heads;
        let n_rep = self.n_heads / self.n_kv_heads;

        let (b_sz, seq_len, _) = input.shape().dims3()?;

        // Reshape and transpose for multi-head attention
        let q = q
            .reshape((b_sz, seq_len, self.n_heads, head_dim))?
            .transpose(1, 2)?; // (b_sz, n_heads, seq_len, head_dim)
        let mut k = k_new
            .reshape((b_sz, seq_len, self.n_kv_heads, head_dim))?
            .transpose(1, 2)?; // (b_sz, n_kv_heads, seq_len, head_dim)
        let mut v = v_new
            .reshape((b_sz, seq_len, self.n_kv_heads, head_dim))?
            .transpose(1, 2)?; // (b_sz, n_kv_heads, seq_len, head_dim)

        // Apply rotary embeddings only to the new k tokens
        let (q, k_rotated) =
            Self::apply_rotary_emb(&q, &k, seq_len, head_dim, self.rope_theta, &self.device)?;
        k = k_rotated;

        // Update cache with new k, v
        let total_seq_len = if let Some(ref k_cached) = kv_cache.k_cache {
            // Concatenate with cached values
            k = Tensor::cat(&[k_cached, &k], 2)?; // Concatenate on seq_len dimension
            k.dim(2)?
        } else {
            seq_len
        };

        if let Some(ref v_cached) = kv_cache.v_cache {
            v = Tensor::cat(&[v_cached, &v], 2)?; // Concatenate on seq_len dimension
        }

        // Update cache
        kv_cache.k_cache = Some(k.clone());
        kv_cache.v_cache = Some(v.clone());


        // Repeat KV heads
        let k = Self::repeat_kv(k, n_rep)?;
        let v = Self::repeat_kv(v, n_rep)?;

        // Scaled dot-product attention
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let attn_scores = q.matmul(&k.transpose(2, 3)?)?.affine(scaling, 0.0)?;

        // Mask for causal attention (only mask the current query positions)
        let mask_tensor = Self::create_causal_mask(total_seq_len, &self.device)?;
        // For cached case, we only look at the last seq_len rows of the mask
        let mask_tensor = if position > 0 {
            mask_tensor.narrow(2, total_seq_len - seq_len, seq_len)?
        } else {
            mask_tensor
        };
        let attn_scores = attn_scores.broadcast_add(&mask_tensor)?;

        // Attention probabilities
        let attn_probs = candle_nn::ops::softmax(&attn_scores, candle_core::D::Minus1)?;

        // Attention output
        let context = attn_probs.matmul(&v)?;
        let context: Tensor = context
            .transpose(1, 2)? // (b_sz, seq_len, n_heads, head_dim)
            .reshape(&[b_sz, seq_len, self.hidden_size])?; // (b_sz, seq_len, hidden_size)

        // Final linear projection
        let output = self.o_proj.forward(&context)?;

        Ok(output)
    }
}

impl Layer for CausalSelfAttentionLayer {
    fn forward(&self, input: &candle_core::Tensor) -> CandleResult<candle_core::Tensor> {
        let mut cache = KVCache::new();
        self.forward_with_cache(input, &mut cache, 0)
    }
}
