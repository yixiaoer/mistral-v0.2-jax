from jax import Array
from transformers.models.mistral.modeling_mistral import MistralModel

from .decoder import DecoderParams, convert_decoder_params, forward_decoder, shard_decoder_params
from .embedding import EmbeddingParams, convert_embedding_params, forward_embedding, shard_embedding_params
from .kvcache import KVCache
from .rms_norm import RMSNormParams, convert_rms_norm_params, forward_rms_norm, shard_rms_norm_params
from .rotary_embedding import RotaryValues

MistralModelParams = tuple[EmbeddingParams, DecoderParams, RMSNormParams]

def convert_mistral_model_params(model: MistralModel) -> MistralModelParams:
    """
    Converts MistralModel (PyTorch tensor) to MistralModelParams(JAX Array).

    Args:
        model (MistralModel): Mistral v0.2 model.

    Returns:
        MistralModelParams: The converted Mistral parameters.
    """
    embedding = convert_embedding_params(model.embed_tokens)
    decoder_layers = convert_decoder_params(model.layers)
    norm = convert_rms_norm_params(model.norm)
    return embedding, decoder_layers, norm

def convert_back_mistral_model_params():
    pass

def shard_mistral_model_params(params: MistralModelParams):
    """
    Shard the MistralModelParams params for distributed computing.

    Args:
        params (MistralModelParams): The Mistral parameters.

    Returns:
        MistralModelParams: The Mistral parameters modified with tensor parallelism, allowing for distributed computation across multiple devices.
    """
    embedding, decoder_layers, norm = params
    embedding = shard_embedding_params(embedding)
    decoder_layers = shard_decoder_params(decoder_layers)
    norm = shard_rms_norm_params(norm)
    return embedding, decoder_layers, norm

def forward_mistral_model(params: MistralModelParams, input_ids: Array, qk_mask: Array, rotary_values: RotaryValues, kv_cache_pre: KVCache) -> tuple[Array, KVCache]:
    """
    Executes the forward pass of mistral model.

    Args:
        params (MistralModelParams): The Mistral model parameters.
        input_ids (Array): The input sequences to the decoder block.
        qk_mask (Array): The qk mask for the attention mechanism, determining which parts of the sequence are allowed to attend to each other.
        rotary_values (RotaryValues): Rotary positional embeddings values.
        kv_cache_pre (KVCache): The previous KVCache.

    Returns:
        tuple[Array, KVCache]: A tuple containing the output sequence after mistral lm, and previous KVCache.
    """
    embedding, decoder_layers, norm = params
    seq = forward_embedding(embedding, input_ids)
    seq, kv_cache_pre = forward_decoder(decoder_layers, seq, qk_mask, rotary_values, kv_cache_pre)
    seq = forward_rms_norm(norm, seq)
    return seq, kv_cache_pre

def test_forward_mistral_model():
    pass
