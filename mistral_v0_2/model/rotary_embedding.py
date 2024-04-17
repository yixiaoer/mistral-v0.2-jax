from typing import NamedTuple

import einops as op
import jax
from jax import Array
import jax.numpy as jnp

# TODO: eliminate this
d_k = 128

# TODO: Mostly taken from https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py
# and https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L92
def _make_weights(seq_len: int, d_k: int) -> tuple[Array, Array]:
    inv_freq = 1. / (1000000 ** (jnp.arange(0, d_k, 2) / d_k))
    sinusoid_inp = op.einsum(jnp.arange(seq_len), inv_freq, 'L, j -> L j')
    sin_val = jnp.sin(sinusoid_inp)
    cos_val = jnp.cos(sinusoid_inp)
    sin_val = op.repeat(sin_val, 'L K -> L (i K)', i=2)
    cos_val = op.repeat(cos_val, 'L K -> L (i K)', i=2)
    return sin_val, cos_val

def _rotate_half(x: Array) -> Array:
    x = op.rearrange(x, '... (i x) -> ... i x', i=2)  # split the last dimension: (..., n) -> (..., 2, n // 2)
    x = x[..., ::-1, :]  # reverse dimension -2
    x = x.at[..., 0, :].multiply(-1)  # negate the first half of dimension -2
    x = op.rearrange(x, '... i x -> ... (i x)')  # merge the last two dimensions: (..., 2, n // 2) -> (..., n)
    return x

class RotaryValues(NamedTuple):
    sin_val: Array
    cos_val: Array

def forward_rotary_embedding(m: Array, *, rotary_values: RotaryValues) -> Array:
    sin_val, cos_val = rotary_values
    assert sin_val.dtype == jnp.float32
    assert cos_val.dtype == jnp.float32
    n = _rotate_half(m)
    a = op.einsum(m, cos_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    b = op.einsum(n, sin_val, 'B ... L K, B L K -> B ... L K').astype(m.dtype)
    return a + b

def make_rotary_values(batch_size: int, seq_len: int) -> RotaryValues:
    """
    Generates sine and cosine values for rotary positional embeddings based on sequence length.

    Args:
        batch_size (int): The number of sequences in a batch.
        seq_len (int): The length of every sequences in a batch.

    Returns:
        RotaryValues: Rotary embedding values with sine values, and cosine values.
    """
    sin_val, cos_val = _make_weights(seq_len, d_k)

    sin_val = jnp.repeat(sin_val[None], batch_size, axis=0)
    cos_val = jnp.repeat(cos_val[None], batch_size, axis=0)
    return RotaryValues(sin_val, cos_val)

def get_rotary_values_at_position(rotary_values: RotaryValues, position: Array) -> RotaryValues:
    """
    Extracts the rotary positional embedding values for a specific position across all sequences in a batch.

    Args:
        rotary_values (RotaryValues): The rotary values from which to extract the positional embeddings.
        position (Array): The position for which to extract the rotary values.

    Returns:
        RotaryValues: Rotary embedding values for the specified position.
    """
    sin_val, cos_val = rotary_values
    sin_val = sin_val[:, position][:, None]
    cos_val = cos_val[:, position][:, None]

    rotary_values = RotaryValues(sin_val, cos_val)
    return rotary_values
