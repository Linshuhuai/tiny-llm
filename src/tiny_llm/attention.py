import mlx.core as mx
from .basics import softmax, linear

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    seq_len = query.shape[-2]
    d_k = query.shape[-1]  # embedding dim of each head
    attn_weight = query @ key.swapaxes(-2, -1)  # only switch last two dim of K, [N.. x Lq x Lk]
    if scale is None:
        scale = 1 / mx.sqrt(d_k)
    attn_weight *= scale
    NEG_INF = -1e9
    attn_bias = mx.zeros_like(attn_weight)
    if mask == "causal":
        attn_bias = mx.triu(mx.ones((attn_weight)), k=1) * NEG_INF  # Upper tri matrix 1->NEG_INF
    elif mask is not None:
        if mask.dtype == mx.bool_:
            attn_bias = (~mask).astype(query.dtype) * NEG_INF  # False->1->NEG_INF, True->0->0
        else:
            attn_bias += mask
    attn_weight += attn_bias
    attn_weight = mx.softmax(attn_weight, -1)  # only apply softmax to last dim (Lk)
    return attn_weight @ value  # [N.. x L x D]

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,  # (H x D) x E
        wk: mx.array,  # (H x D) x E
        wv: mx.array,  # (H x D) x E
        wo: mx.array,  # E x (H x D)
    ):
        self.d_k = hidden_size // num_heads
        self.h = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        *prefix, seq_len, hidden_size = query.shape  # reshape x for attention calculation convenience
        assert(self.d_k == hidden_size // self.h)  # assert E = H x D
        query = linear(query, self.wq) # [N.., L, E] @ [E, (H x D)] -> [N.., L, (H x D)]
        query = query.reshape(*prefix, seq_len, self.h, self.d_k)  # [N.., L, (H x D)] -> [N.., L, H, D]
        query = query.swapaxes(-3, -2)  # [N.., L, H, D] -> [N.., H, L, D]
        key = linear(key, self.wk)
        key = key.reshape(*prefix, seq_len, self.h, self.d_k)
        key = key.swapaxes(-3, -2)
        value = linear(value, self.wv)
        value = value.reshape(*prefix, seq_len, self.h, self.d_k)
        value = value.swapaxes(-3, -2)
        self.attn = scaled_dot_product_attention_simple(query=query, key=key, value=value, mask=mask)
        self.attn = self.attn.swapaxes(-3, -2)  # [N.., H, L, D] -> [N.., L, H, D]
        self.attn = self.attn.reshape(*prefix, seq_len, hidden_size)  # [N.., L, H, D] -> [N.., L, (H x D)]
        return linear(self.attn, self.wo)  # [N.., L, (H x D)] @ [(H x D), E] -> [N.., L, E]

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
