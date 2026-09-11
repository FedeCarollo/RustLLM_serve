use std::sync::RwLock;
use crate::llm::layer::Layer;
use crate::llm::linear::LinearLayer;
use candle_core::{DType, Device, Tensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::Result as CandleResult;

/// KV cache for storing key and value tensors across generation steps
#[derive(Clone)]
pub struct KVCache {
    pub k_cache: Option<Tensor>,
    pub v_cache: Option<Tensor>,
    pub current_seq_len: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            current_seq_len: 0,
            max_seq_len,
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
    d_model: usize,
    dk: usize,
    dk_sqrt: f32,
    causal_mask: Option<Tensor>,
    kv_cache: RwLock<KVCache>,
    cos: Tensor,
    sin: Tensor,
}

impl CausalSelfAttentionLayer {
    pub fn new(
        weights: &MmapedSafetensors,
        prefix: &str,
        n_heads: usize,
        n_kv_heads: usize,
        d_model: usize,
        rope_theta: f32,
        device: Device,
        dtype: DType,
        max_seq_len: usize,
    ) -> CandleResult<Self> {
        let q_proj = LinearLayer::new(weights, &format!("{}.q_proj", prefix), device.clone(), dtype)?;
        let k_proj = LinearLayer::new(weights, &format!("{}.k_proj", prefix), device.clone(), dtype)?;
        let v_proj = LinearLayer::new(weights, &format!("{}.v_proj", prefix), device.clone(), dtype)?;
        let o_proj = LinearLayer::new(weights, &format!("{}.o_proj", prefix), device.clone(), dtype)?;
        let causal_mask = Some(Self::create_causal_mask(max_seq_len, &device, dtype)?);
        let kv_cache = RwLock::new(KVCache::new(max_seq_len));
        let dk = d_model / n_heads;
        let dk_sqrt = (dk as f32).sqrt();
        let (cos, sin) = Self::precompute_frequencies(max_seq_len, rope_theta, dk, &device, dtype)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
            d_model,
            causal_mask,
            kv_cache,
            dk,
            dk_sqrt,
            cos,
            sin,
        })
    }

    fn precompute_frequencies(
        max_seq_len: usize,
        rope_theta: f32,
        head_dim: usize,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<(Tensor, Tensor)> {
        let half_dim = head_dim / 2;

        // inv_freq[i] = 1 / theta^(2i/d) = exp(-2i/d * ln(theta))
        // computed in f32 for numerical precision
        let inv_freq = Tensor::arange(0f32, half_dim as f32, device)?
            .affine(-2.0 * (rope_theta as f64).ln() / head_dim as f64, 0.0)? // -2i/d * ln(theta)
            .exp()?; // (half_dim,) in f32

        // absolute position indices [0, 1, ..., max_seq_len - 1]
        let positions = Tensor::arange(0f32, max_seq_len as f32, device)?; // (max_seq_len,) in f32

        // freqs[t, i] = t * inv_freq[i]
        let freqs = positions
            .unsqueeze(1)?                            // (max_seq_len, 1)
            .broadcast_mul(&inv_freq.unsqueeze(0)?)?; // (max_seq_len, half_dim)

        // cast to model dtype only at the end, after f32 trig ops
        let cos = freqs.cos()?.to_dtype(dtype)?; // (max_seq_len, half_dim)
        let sin = freqs.sin()?.to_dtype(dtype)?; // (max_seq_len, half_dim)

        Ok((cos, sin))
    }

    fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
        let half = x.dim(candle_core::D::Minus1)? / 2;
        let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;    // [..., half_dim]
        let x2 = x.narrow(candle_core::D::Minus1, half, half)?; // [..., half_dim]
        Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1) // [-x2, x1]
    }

    fn apply_rotary_emb(
        &self,
        q: &Tensor,       // [B, n_heads, T, dk]
        k: &Tensor,       // [B, n_kv_heads, T, dk]
        position_offset: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;

        // slice the relevant positions from the precomputed cache
        let cos = self.cos.narrow(0, position_offset, seq_len)?; // [T, half_dim]
        let sin = self.sin.narrow(0, position_offset, seq_len)?; // [T, half_dim]

        // duplicate along last dim to cover full head_dim: [T, half_dim] -> [T, head_dim]
        let cos = Tensor::cat(&[&cos, &cos], 1)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, T, head_dim]
        let sin = Tensor::cat(&[&sin, &sin], 1)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, T, head_dim]

        // x' = x * cos + rotate_half(x) * sin
        let q_rot = (q.broadcast_mul(&cos)? + Self::rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_rot = (k.broadcast_mul(&cos)? + Self::rotate_half(k)?.broadcast_mul(&sin)?)?;

        Ok((q_rot, k_rot))
    }

    fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
        // x: [B, n_kv_heads, T, dk]
        if n_rep == 1 { return Ok(x); }

        let (b, n_kv_heads, t, dk) = x.dims4()?;
        x.unsqueeze(2)?                                   // [B, n_kv_heads, 1, T, dk]
         .expand(&[b, n_kv_heads, n_rep, t, dk])?        // [B, n_kv_heads, n_rep, T, dk]
         .reshape((b, n_kv_heads * n_rep, t, dk))        // [B, n_heads, T, dk]
    }

    fn create_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> CandleResult<Tensor> {
        let rows = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(1)?; // (seq_len, 1)
        let cols = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(0)?; // (1, seq_len)

        let mask = rows.broadcast_ge(&cols)?;

        let on_true  = Tensor::zeros((1,), dtype, device)?;
        let on_false = Tensor::new(&[f32::NEG_INFINITY], device)?.to_dtype(dtype)?;

        mask.where_cond(
            &on_true.broadcast_as(mask.shape())?,
            &on_false.broadcast_as(mask.shape())?,
        )?
        .unsqueeze(0)? // (1, seq_len, seq_len)
        .unsqueeze(0)  // (1, 1, seq_len, seq_len)
    }
}

impl CausalSelfAttentionLayer {
    /// Forward pass with KV cache support
    pub fn forward_with_cache(
        &self,
        input: &Tensor,
    ) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _) = input.dims3()?;

        // linear projections
        let q = self.q_proj.forward(input)?; // [B, T, n_heads * dk]
        let k = self.k_proj.forward(input)?; // [B, T, n_kv_heads * dk]
        let v = self.v_proj.forward(input)?; // [B, T, n_kv_heads * dk]

        // reshape and transpose -> [B, n_heads, T, dk]
        let q = q.reshape((batch_size, seq_len, self.n_heads, self.dk))?.transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.n_kv_heads, self.dk))?.transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.n_kv_heads, self.dk))?.transpose(1, 2)?;

        // position offset = how many tokens are already in the cache
        let position_offset = self.kv_cache.read().unwrap().current_seq_len;

        // apply rotary embeddings with correct absolute positions
        let (q, k) = self.apply_rotary_emb(&q, &k, position_offset)?;

        // update cache in-place
        let (active_k, active_v, total_seq_len) = {
            let mut cache = self.kv_cache.write().unwrap();

            if cache.k_cache.is_none() {
                // Pre-allocate the full max_seq_len tensor when first token arrives
                cache.k_cache = Some(Tensor::zeros((batch_size, self.n_kv_heads, cache.max_seq_len, self.dk), k.dtype(), k.device())?);
                cache.v_cache = Some(Tensor::zeros((batch_size, self.n_kv_heads, cache.max_seq_len, self.dk), v.dtype(), v.device())?);
            }
            
            cache.current_seq_len += seq_len;
            let total_seq_len = cache.current_seq_len;

            let k_cache = cache.k_cache.as_ref().unwrap();
            let v_cache = cache.v_cache.as_ref().unwrap();

            k_cache.slice_set(&k.contiguous()?, 2, position_offset)?;
            v_cache.slice_set(&v.contiguous()?, 2, position_offset)?;

            let active_k = k_cache.narrow(2, 0, total_seq_len)?;
            let active_v = v_cache.narrow(2, 0, total_seq_len)?;
            
            (active_k, active_v, total_seq_len)
        };

        // expand kv heads to match q heads (GQA / MQA support)
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = Self::repeat_kv(active_k, n_rep)?; // [B, n_heads, total_T, dk]
        let v = Self::repeat_kv(active_v, n_rep)?; // [B, n_heads, total_T, dk]

        // scaled dot-product attention: scale q *before* matmul to avoid overflow
        let scale = 1.0 / self.dk_sqrt as f64;
        let attn = q.affine(scale, 0.0)?.matmul(&k.transpose(2, 3)?)?; // [B, n_heads, T, total_T]

        // apply causal mask: slice rows [position_offset..position_offset+seq_len]
        let mask = self.causal_mask.as_ref().unwrap()
            .narrow(2, position_offset, seq_len)?
            .narrow(3, 0, total_seq_len)?; // [1, 1, T, total_T]
        let attn = attn.broadcast_add(&mask)?;

        // softmax and weighted sum
        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        let out = attn.matmul(&v)?; // [B, n_heads, T, dk]

        // reassemble heads: [B, n_heads, T, dk] -> [B, T, d_model]
        let out = out
            .transpose(1, 2)?                                          // [B, T, n_heads, dk]
            .reshape((batch_size, seq_len, self.d_model))?;        // [B, T, D]

        self.o_proj.forward(&out)
    }
    pub fn clear_cache(&self) {
        let mut cache = self.kv_cache.write().unwrap();
        // we keep the pre-allocated tensors, just reset the pointer
        cache.current_seq_len = 0;
    }

    pub fn get_cache_len(&self) -> usize {
        self.kv_cache.read().unwrap().current_seq_len
    }
}

impl Layer for CausalSelfAttentionLayer {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len, _) = input.dims3()?;

        // linear projections
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // reshape and transpose -> [B, n_heads, T, dk]
        let q = q.reshape((batch_size, seq_len, self.n_heads, self.dk))?.transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.n_kv_heads, self.dk))?.transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.n_kv_heads, self.dk))?.transpose(1, 2)?;

        // no cache -> position offset is 0
        let (q, k) = self.apply_rotary_emb(&q, &k, 0)?;

        // expand kv heads
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = Self::repeat_kv(k, n_rep)?;
        let v = Self::repeat_kv(v, n_rep)?;

        // attention
        let scale = 1.0 / self.dk_sqrt as f64;
        let attn = q.affine(scale, 0.0)?.matmul(&k.transpose(2, 3)?)?;

        let mask = self.causal_mask.as_ref().unwrap()
            .narrow(2, 0, seq_len)?
            .narrow(3, 0, seq_len)?;
        let attn = attn.broadcast_add(&mask)?;

        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;
        let out = attn.matmul(&v)?;

        // reassemble heads
        let out = out
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.d_model))?;

        self.o_proj.forward(&out)
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use crate::llm::causal_self_attention::*;

    #[test]
    fn test_kv_cache_initialization() -> anyhow::Result<()> {
        let max_seq_len = 100;
        let mut cache = KVCache::new(max_seq_len);
        
        assert!(cache.k_cache.is_none());
        assert!(cache.v_cache.is_none());
        assert_eq!(cache.current_seq_len, 0);
        assert_eq!(cache.max_seq_len, max_seq_len);
        
        // Test clear
        cache.current_seq_len = 10;
        cache.clear();
        assert_eq!(cache.current_seq_len, 0);
        
        Ok(())
    }
}
