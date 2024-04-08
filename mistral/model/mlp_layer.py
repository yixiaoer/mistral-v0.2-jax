import jax
from jax import Array
import jax.numpy as jnp
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralMLP

from ..lib.array_conversion import pt2jax
from ..lib.einshard import einshard

MLPLayerParams = tuple[Array, Array, Array]

def convert_mlp_layer_params(mlp_layer: MistralMLP) -> MLPLayerParams:
    """
    Converts PyTorch MLP layer parameters to a MLPLayerParams compatible with JAX.

    Args:
        mlp_layer (MistralMLP): The PyTorch MLP layer from which to extract the weights.

    Returns:
        MLPLayerParams: The embedding parameters extracted from the PyTorch layer and formatted for compatibility with JAX operations.
    """
    gate_proj = pt2jax(mlp_layer.gate_proj.weight.data.T)
    up_proj = pt2jax(mlp_layer.up_proj.weight.data.T)
    down_proj = pt2jax(mlp_layer.down_proj.weight.data.T)
    return gate_proj, up_proj, down_proj

def convert_back_mlp_layer_params(mlp_layer: MLPLayerParams) -> MistralMLP:
    # mlp_layer_pt = MistralMLP(config_pt)  # TODO: handle config
    pass

def shard_mlp_layer_params(params: MLPLayerParams) -> MLPLayerParams:
    """
    Shard the MLPLayerParams params for distributed computing.

    Args:
        params (MLPLayerParams): The MLPLayerParams parameters.

    Returns:
        MLPLayerParams: The decoder embedding parameters replica for distributed computation across multiple devices.
    """
    gate_proj, up_proj, down_proj = params
    gate_proj = einshard(gate_proj, 'm f -> m f*')
    up_proj = einshard(up_proj, 'm f -> m f*')
    down_proj = einshard(down_proj, 'f m -> f* m')
    return gate_proj, up_proj, down_proj

def forward_mlp_layer(params: MLPLayerParams, seq: Array) -> Array:
    """
    Executes the forward pass of MLP.

    Args:
        params (MLPLayerParams): The MLP layer parameters.
        seq (Array): The input sequences.

    Returns:
        Array: The output after MLP layer.
    """
    gate_proj, up_proj, down_proj = params

    ff = jax.nn.silu(seq @ gate_proj) * (seq @ up_proj)
    seq = ff @ down_proj
    return seq

def test_forward_mlp_layer(model: MistralForCausalLM) -> None:
    pass
