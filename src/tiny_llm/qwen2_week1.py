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
        self.hidden_size = hidden_size
        self.input_layer_norm = RMSNorm(
            dim=self.hidden_size,
            weight=w_input_layernorm,
            eps=rms_norm_eps
        )
        self.post_attention_layer_norm = RMSNorm(
            dim=self.hidden_size,
            weight=w_post_attention_layernorm,
            eps=rms_norm_eps
        )
        self.attention = Qwen2MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta
        )
        self.mlp = Qwen2MLP(
            dim=self.hidden_size,
            hidden_dim=intermediate_size,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down
        )
        

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.input_layer_norm(x)  # input_layernorm (RMSNorm)
        r = self.attention(r, mask)  # Qwen2MultiHeadAttention
        h = x + r  # residual
        r = self.post_attention_layer_norm(h)  # post_attention_layernorm (RMSNorm)
        r = self.mlp(r)  # MLP
        output = h + r  # residual
        return output

class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.as_linear = mlx_model.args.tie_word_embeddings
        
        precision = mx.float16
        self.embedding = Embedding(
            vocab_size=mlx_model.args.vocab_size,
            embedding_dim=mlx_model.args.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision),
        )
        self.layers = []
        for i in range(self.num_hidden_layers):
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj)
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj)

            layer = Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq.astype(precision),
                wk=wk.astype(precision),
                wv=wv.astype(precision),
                wo=wo.astype(precision),
                bq=mlx_model.model.layers[i].self_attn.q_proj.bias.astype(precision),
                bk=mlx_model.model.layers[i].self_attn.k_proj.bias.astype(precision),
                bv=mlx_model.model.layers[i].self_attn.v_proj.bias.astype(precision),
                w_gate=w_gate.astype(precision),
                w_up=w_up.astype(precision),
                w_down=w_down.astype(precision),
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight.astype(precision),
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight.astype(precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers.append(layer)
        self.last_layer_norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        x = self.embedding(inputs)
        for i in range(self.num_hidden_layers):
            x = self.layers[i](x, mask="causal")
        x = self.last_layer_norm(x)
        if self.as_linear:
            output = self.embedding.as_linear(x)
        else:
            output = linear(x, self.w_lm_head)
        return output
