import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight.astype(mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)  # [N.., D]
        *prefix, dim = x.shape
        assert(dim == self.dim)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True)) + self.eps  # only calculate last dim, keep other dims
        return (x / rms) * self.weight  # [N.., D]
        
