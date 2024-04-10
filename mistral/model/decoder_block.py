from jax import Array
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from .attention import AttentionParams, convert_attention_params, forward_attention, shard_attention_params
from .kvcache import KVCache
from .mlp_layer import MLPLayerParams, convert_mlp_layer_params, forward_mlp_layer, shard_mlp_layer_params
from .rms_norm import RMSNormParams, convert_rms_norm_params, forward_rms_norm, shard_rms_norm_params
from .rotary_embedding import RotaryValues

DecoderBlockParams = tuple[RMSNormParams, AttentionParams, MLPLayerParams, RMSNormParams]

def convert_decoder_block_params(decoder_block: MistralDecoderLayer) -> DecoderBlockParams:
    """
    Converts decoder block parameters from MistralDecoderLayer(PyTorch tensor) to DecoderBlockParams(JAX Array).

    Args:
        decoder_block (MistralDecoderLayer): The decoder block's MistralDecoderLayer.

    Returns:
        DecoderBlockParams: The converted decoder block parameters.
    """

    input_layernorm = convert_rms_norm_params(decoder_block.input_layernorm)
    self_attn = convert_attention_params(decoder_block.self_attn)
    mlp = convert_mlp_layer_params(decoder_block.mlp)
    post_attention_layernorm = convert_rms_norm_params(decoder_block.post_attention_layernorm)
    return input_layernorm, self_attn, mlp, post_attention_layernorm

def convert_back_decoder_block_params():
    pass

def shard_decoder_block_params(params: DecoderBlockParams) -> DecoderBlockParams:
    """
    Shard the DecoderBlockParams params for distributed computing.

    Args:
        params (DecoderBlockParams): The decoder block parameters.

    Returns:
        DecoderBlockParams: The decoder block parameters modified with tensor parallelism, allowing for distributed computation across multiple devices.
    """
    input_layernorm, self_attn, mlp, post_attention_layernorm = params
    input_layernorm = shard_rms_norm_params(input_layernorm)
    self_attn = shard_attention_params(self_attn)
    mlp = shard_mlp_layer_params(mlp)
    post_attention_layernorm = shard_rms_norm_params(post_attention_layernorm)
    return input_layernorm, self_attn, mlp, post_attention_layernorm

def forward_decoder_block(params: DecoderBlockParams, seq: Array, qk_mask: Array, rotary_values: RotaryValues ,kv_cache_cur: KVCache, kv_cache_pre: KVCache) -> tuple[Array, KVCache, KVCache]:
    """
    Executes the forward pass of a decoder block using the specified parameters and input sequence.

    Args:
        params (DecoderBlockParams): The decoder block parameters.
        seq (Array): The input sequences to the decoder block.
        qk_mask (Array): The qk mask for the attention mechanism, determining which parts of the sequence are allowed to attend to each other.
        rotary_values (RotaryValues): Rotary positional embeddings values.
        kv_cache_cur (KVCache): The current KVCache.
        kv_cache_pre (KVCache): The previous KVCache.

    Returns:
        tuple[Array, KVCache, KVCache]: A tuple containing the output sequence after decoder block, and the updated current and previous KVCache.
    """
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
