import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    # x [N.. x I], w [O x I], bias [O], I is input dim and O is output dim.
    if bias is None:
        return x @ w.swapaxes(-2, -1)
    return x @ w.swapaxes(-2, -1) + bias

def silu(x: mx.array) -> mx.array:
    pass
