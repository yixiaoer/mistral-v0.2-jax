from jax import Array
import jax.numpy as jnp
import torch
from torch.nn import Embedding as TorchEmbedding
from transformers import MistralForCausalLM

from ..lib.array_conversion import pt2jax
from ..lib.einshard import einshard

EmbeddingParams = Array

def convert_embedding_params(embedding: TorchEmbedding) -> EmbeddingParams:
    return pt2jax(embedding.weight.data)

def convert_back_embedding_params():
    pass

def shard_embedding_params(params: EmbeddingParams) -> EmbeddingParams:
    return einshard(params, '... -> 1 ...')

def forward_embedding(params: EmbeddingParams, input_ids: Array) -> Array:
    return params[input_ids]

def test_forward_embedding(model: MistralForCausalLM) -> None:
    embedding_pt = model.model.embed_tokens
    input_ids_pt = torch.tensor([1, 20, 3, 5, 2, 7], dtype=torch.int32)
    result_pt = embedding_pt(input_ids_pt)
    result_pt_to_jax = pt2jax(result_pt)

    params = convert_embedding_params(embedding_pt)
    input_ids_jax = pt2jax(input_ids_pt)
    result_jax = forward_embedding(params, input_ids_jax)

    assert jnp.allclose(result_pt_to_jax, result_jax, atol=1e-5)
