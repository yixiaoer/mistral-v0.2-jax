from jax import Array
import jax.numpy as jnp
import torch
from torch.nn import Embedding as TorchEmbedding
from transformers import MistralForCausalLM

from ..lib.array_conversion import pt2jax
from ..lib.einshard import einshard

EmbeddingParams = Array

def convert_embedding_params(embedding: TorchEmbedding) -> EmbeddingParams:
    """
    Converts PyTorch embedding parameters to a EmbeddingParams compatible with JAX.

    Args:
        embedding (TorchEmbedding): The PyTorch embedding layer from which to extract the weights.

    Returns:
        EmbeddingParams: The embedding parameters extracted from the PyTorch layer and formatted for compatibility with JAX operations.
    """
    return pt2jax(embedding.weight.data)

def convert_back_embedding_params():
    pass

def shard_embedding_params(params: EmbeddingParams) -> EmbeddingParams:
    """
    Shard the EmbeddingParams params for distributed computing.

    Args:
        params (EmbeddingParams): The EmbeddingParams parameters.

    Returns:
        EmbeddingParams: The decoder embedding parameters replica for distributed computation across multiple devices.
    """
    return einshard(params, '... -> 1 ...')

def forward_embedding(params: EmbeddingParams, input_ids: Array) -> Array:
    """
    Get the embedding with input IDS.

    Args:
        params (EmbeddingParams): The embedding parameters.
        input_ids (Array): An array of input IDS to look up the embedding.

    Returns:
        Array: The embedding Array of input IDS.
    """
    return params[input_ids]

def test_forward_embedding(model: MistralForCausalLM) -> None:
    """
    Tests the embedding parameters.

    Args:
        model (MistralForCausalLM): PyTorch Mistral model to compare with this implementation.

    Returns:
        None.
    """
    embedding_pt = model.model.embed_tokens
    input_ids_pt = torch.tensor([1, 20, 3, 5, 2, 7], dtype=torch.int32)
    result_pt = embedding_pt(input_ids_pt)
    result_pt_to_jax = pt2jax(result_pt)

    params = convert_embedding_params(embedding_pt)
    input_ids_jax = pt2jax(input_ids_pt)
    result_jax = forward_embedding(params, input_ids_jax)

    assert jnp.allclose(result_pt_to_jax, result_jax, atol=1e-5)
