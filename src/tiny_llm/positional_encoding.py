import mlx.core as mx

class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.max_seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.cos_freqs, self.sin_freqs = self.build_cos_sin_freqs()
    
    def build_cos_sin_freqs(self):
        assert(self.dims % 2 == 0)
        half_dim = self.dims // 2
        pos = mx.arange(self.max_seq_len)  # positions: [MAX_SEQ_LEN]
        i = mx.arange(half_dim)
        inv_freq = mx.power(self.base, -2.0 * i / self.dims) # inv_freq: [D//2]
        freqs = pos[:, None] * inv_freq[None, :]  # [MAX_SEQ_LEN, 1] @ [1, D//2] -> [MAX_SEQ_LEN, D//2]
        return mx.cos(freqs), mx.sin(freqs)  # return cos_freqs, sin_freqs

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        *prefix, seq_len, head_nums, d_k = x.shape
        assert(d_k == self.dims)
        
        if offset is None: 
            cos_freqs = self.cos_freqs[:seq_len]  # [MAX_SEQ_LEN, D//2] -> [L, D//2]
            sin_freqs = self.sin_freqs[:seq_len]
            cos_freqs = cos_freqs[None, :, None, :]  # [1, L, 1, D//2]
            sin_freqs = sin_freqs[None, :, None, :]
        elif isinstance(offset, slice):
            cos_freqs = self.cos_freqs[offset]  # [MAX_SEQ_LEN, D//2] -> [L, D//2]
            sin_freqs = self.sin_freqs[offset]
            cos_freqs = cos_freqs[None, :, None, :]  # [1, L, 1, D//2]
            sin_freqs = sin_freqs[None, :, None, :]
        elif isinstance(offset, list):
            pass
        
        if self.traditional:  # RoPE in the traditional form, assume
            x = x.reshape(*prefix, seq_len, head_nums, d_k//2, 2)  # reshape x to [N.., L, H, D//2, 2]
            x_even = x[..., 0]  # [N.., L, H, D//2]
            x_odd = x[..., 1]
            output_even = x_even * cos_freqs - x_odd * sin_freqs  # [N.., L, H, D//2]
            output_odd = x_even * sin_freqs + x_odd * cos_freqs
            output = mx.stack([output_even, output_odd], axis=-1)  # [N.., L, H, D//2, 2]
        else:
            half_dim = d_k // 2
            x_first = x[..., :half_dim]  # [N.., L, H, D//2]
            x_second = x[..., half_dim:]
            output_first = x_first * cos_freqs - x_second * sin_freqs  # [N.., L, H, D//2]
            output_second = x_first * sin_freqs + x_second * cos_freqs
            output = mx.concatenate([output_first, output_second], axis=-1)  # [N.., L, H, D//2, 2]
        return output.reshape(*prefix, seq_len, head_nums, d_k)  # reshape output to [N.., L, H, D]
            
            
            
