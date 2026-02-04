import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.base = theta

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        *prefix, seq_len, emb_dim = x.shape
        q = linear(x, self.wq, self.bq)
        q = q.reshape(*prefix, seq_len, self.num_heads, self.hidden_size // self.num_heads)  # [B, L, H_q, D]
        k = linear(x, self.wk, self.bk)
        k = k.reshape(*prefix, seq_len, self.num_kv_heads, self.hidden_size // self.num_heads)  # [B, L, H, D]
        v = linear(x, self.wv, self.bv)
        v = v.reshape(*prefix, seq_len, self.num_kv_heads, self.hidden_size // self.num_heads)  # [B, L, H, D]

        rope = RoPE(
            dims = self.hidden_size // self.num_heads, 
            seq_len = self.max_seq_len, 
            base = self.base,
            traditional = False
        )
        q = rope(q, offset=slice(0, seq_len))  # [B, L, H_q, D]
        q = q.swapaxes(-3, -2)  # [B, H_q, L, D]
        k = rope(k, offset=slice(0, seq_len))  # [B, L, H, D]
        k = k.swapaxes(-3, -2)  # [B, H, L, D]
        v = v.swapaxes(-3, -2)
        x = scaled_dot_product_attention_grouped(query=q, key=k, value=v, mask=mask)  # [B, H_q, L, D]
        x = x.swapaxes(-3, -2)  # [B, L, H_q, D]
        x = x.reshape(*prefix, seq_len, emb_dim) # [B, L, E]
        return linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate  # [I x E]
        self.w_up = w_up  # [I x E]
        self.w_down = w_down  # [E x I]      

    def __call__(self, x: mx.array) -> mx.array:
        gate = linear(x, self.w_gate)  #[N.. x L x E] -> [N.. x L x I]
        up = linear(x, self.w_up)  # [N.. x L x I]
        gated = silu(gate) * up  # element-wise multiplication, [N.. x L x I]
        return linear(gated, self.w_down)  # [N.. x L x E]


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
