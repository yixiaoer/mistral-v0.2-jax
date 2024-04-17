from jax import Array
from torch.nn import ModuleList as TorchModuleList

from .decoder_block import DecoderBlockParams, convert_decoder_block_params, forward_decoder_block, shard_decoder_block_params
from .kvcache import KVCache
from .rotary_embedding import RotaryValues

DecoderParams = list[DecoderBlockParams]

def convert_decoder_params(layers: TorchModuleList) -> DecoderParams:
    """
    Converts decoder TorchModuleList layers(PyTorch tensor) to DecoderParams(JAX Array).

    Args:
        layers (TorchModuleList): Layers.

    Returns:
        DecoderParams: The converted decoder parameters.
    """
    return [convert_decoder_block_params(layer) for layer in layers]

def convert_back_decoder_params():
    raise NotImplementedError

def shard_decoder_params(layers: DecoderParams) -> DecoderParams:
    """
    Shard the DecoderParams params for distributed computing.

    Args:
        params (DecoderParams): The decoder parameters.

    Returns:
        DecoderParams: The decoder parameters modified with tensor parallelism, allowing for distributed computation across multiple devices.
    """
    return [shard_decoder_block_params(layer) for layer in layers]

def forward_decoder(params: DecoderParams, seq: Array, qk_mask: Array, rotary_values: RotaryValues, kv_cache_pre: KVCache) -> tuple[Array, KVCache]:
    """
    Executes the forward pass of all decoder blocks.

    Args:
        params (DecoderParams): The decoder parameters.
        seq (Array): The input sequences to the decoder block.
        qk_mask (Array): The qk mask for the attention mechanism, determining which parts of the sequence are allowed to attend to each other.
        rotary_values (RotaryValues): Rotary positional embeddings values.
        kv_cache_cur (KVCache): The current KVCache.
        kv_cache_pre (KVCache): The previous KVCache.

    Returns:
        tuple[Array, KVCache]: A tuple containing the output sequence after all decoder blocks, and previous KVCache.
    """
    # TODO: jax.lax.scan
    kv_cache_cur = None
    for param in params:
        seq, kv_cache_cur, kv_cache_pre = forward_decoder_block(param, seq, qk_mask, rotary_values, kv_cache_cur, kv_cache_pre)
    kv_cache_pre = kv_cache_cur
    return seq, kv_cache_pre

def test_forward_decoder():
    pass
