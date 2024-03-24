import jax
from jax import Array
import jax.numpy as jnp
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralMLP

from ..lib.array_conversion import pt2jax
from ..lib.einshard import einshard

MLPLayerParams = tuple[Array, Array, Array]

def convert_mlp_layer_params(mlp_layer: MistralMLP) -> MLPLayerParams:
    gate_proj = pt2jax(mlp_layer.gate_proj.weight.data.T)
    up_proj = pt2jax(mlp_layer.up_proj.weight.data.T)
    down_proj = pt2jax(mlp_layer.down_proj.weight.data.T)
    return gate_proj, up_proj, down_proj

def convert_back_mlp_layer_params(mlp_layer: MLPLayerParams) -> MistralMLP:
    # mlp_layer_pt = MistralMLP(config_pt)  # TODO: handle config
    pass

def shard_mlp_layer_params(params: MLPLayerParams) -> MLPLayerParams:
    gate_proj, up_proj, down_proj = params
    gate_proj = einshard(gate_proj, 'm f -> m f1')
    up_proj = einshard(up_proj, 'm f -> m f1')
    down_proj = einshard(down_proj, 'f m -> f1 m')
    return gate_proj, up_proj, down_proj

def forward_mlp_layer(params: MLPLayerParams, seq: Array) -> Array:
    gate_proj, up_proj, down_proj = params

    ff = jax.nn.silu(seq @ gate_proj) * (seq @ up_proj)
    seq = ff @ down_proj
    return seq

def test_forward_mlp_layer(model: MistralForCausalLM) -> None:
    pass
