from jax import Array
from torch.nn import ModuleList as TorchModuleList

from .decoder_block import DecoderBlockParams, convert_decoder_block_params, forward_decoder_block, shard_decoder_block_params
from .kvcache import KVCache
from .rotary_embedding import RotaryValues
from ..lib.einshard import einshard

DecoderParams = list[DecoderBlockParams]

def convert_decoder_params(layers: TorchModuleList) -> DecoderParams:
    return [convert_decoder_block_params(layer) for layer in layers]

def convert_back_decoder_params():
    raise NotImplementedError

def shard_decoder_params(layers: DecoderParams) -> DecoderParams:
    return [shard_decoder_block_params(layer) for layer in layers]

def forward_decoder(params: DecoderParams, seq: Array, qk_mask: Array, rotary_values: RotaryValues, kv_cache_pre: KVCache) -> tuple[Array, KVCache]:
    # TODO: jax.lax.scan
    kv_cache_cur = None
    for param in params:
        seq, kv_cache_cur, kv_cache_pre = forward_decoder_block(param, seq, qk_mask, rotary_values, kv_cache_cur, kv_cache_pre)
    kv_cache_pre = kv_cache_cur
    return seq, kv_cache_pre

def test_forward_decoder():
    pass
