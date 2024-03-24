from jax import Array
import jax.numpy as jnp
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from .attention import AttentionParams, convert_attention_params, forward_attention, shard_attention_params
from .kvcache import KVCache
from .mlp_layer import MLPLayerParams, convert_mlp_layer_params, forward_mlp_layer, shard_mlp_layer_params
from .rms_norm import RMSNormParams, convert_rms_norm_params, forward_rms_norm, shard_rms_norm_params
from .rotary_embedding import RotaryValues
from ..lib.einshard import einshard

DecoderBlockParams = tuple[RMSNormParams, AttentionParams, MLPLayerParams, RMSNormParams]

def convert_decoder_block_params(decoder_block: MistralDecoderLayer) -> DecoderBlockParams:
    input_layernorm = convert_rms_norm_params(decoder_block.input_layernorm)
    self_attn = convert_attention_params(decoder_block.self_attn)
    mlp = convert_mlp_layer_params(decoder_block.mlp)
    post_attention_layernorm = convert_rms_norm_params(decoder_block.post_attention_layernorm)
    return input_layernorm, self_attn, mlp, post_attention_layernorm

def convert_back_decoder_block_params():
    pass

def shard_decoder_block_params(params: DecoderBlockParams) -> DecoderBlockParams:
    input_layernorm, self_attn, mlp, post_attention_layernorm = params
    input_layernorm = shard_rms_norm_params(input_layernorm)
    self_attn = shard_attention_params(self_attn)
    mlp = shard_mlp_layer_params(mlp)
    post_attention_layernorm = shard_rms_norm_params(post_attention_layernorm)
    return input_layernorm, self_attn, mlp, post_attention_layernorm

def forward_decoder_block(params: DecoderBlockParams, seq: Array, qk_mask: Array, rotary_values: RotaryValues ,kv_cache_cur: KVCache, kv_cache_pre: KVCache) -> tuple[Array, KVCache, KVCache]:
    input_layernorm, self_attn, mlp, post_attention_layernorm = params

    # residual connection
    seq_ = seq
    seq = forward_rms_norm(input_layernorm, seq)
    seq, kv_cache_cur, kv_cache_pre = forward_attention(self_attn, seq, qk_mask, rotary_values, kv_cache_cur, kv_cache_pre)
    seq += seq_

    seq_ = seq
    seq = forward_rms_norm(post_attention_layernorm, seq)
    seq = forward_mlp_layer(mlp, seq)
    seq += seq_
    return seq, kv_cache_cur, kv_cache_pre

def test_forward_decoder_block(model: MistralForCausalLM) -> None:
    pass
