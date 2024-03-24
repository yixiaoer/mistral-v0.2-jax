import jax
from jax import Array
import jax.numpy as jnp
import torch
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from ..lib.array_conversion import jax2pt, pt2jax
from ..lib.einshard import einshard

# TODO: eliminate this
d_model = 4096
rms_norm_eps = 1e-5

RMSNormParams = Array

def convert_rms_norm_params(rms_norm: MistralRMSNorm) -> RMSNormParams:
    return pt2jax(rms_norm.weight)

def convert_back_rms_norm_params(rms_norm: RMSNormParams) -> MistralRMSNorm:
    rms_norm_pt = MistralRMSNorm(rms_norm.shape[0], rms_norm_eps)
    rms_norm_pt.weight = torch.nn.Parameter(jax2pt(rms_norm))
    return rms_norm_pt

def shard_rms_norm_params(params: RMSNormParams) -> RMSNormParams:
    return einshard(params, '... -> 1 ...')

# Taken from https://github.com/ayaka14732/llama-2-jax/blob/main/lib/llama/rms_norm.py
def forward_rms_norm(params: RMSNormParams, x: Array) -> Array:
    x_rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + rms_norm_eps)
    y = x / x_rms * params
    return y

def test_forward_rms_norm(model: MistralForCausalLM) -> None:
    d_model = 4096

    seq_pt = torch.rand(2, 14, d_model)
    rms_norm = model.model.norm
    out_pt = rms_norm(seq_pt)
    out_pt_to_jax = pt2jax(out_pt)

    seq_jax = pt2jax(seq_pt)
    params = convert_rms_norm_params(rms_norm)
    out_jax = forward_rms_norm(params, seq_jax)

    assert jnp.allclose(out_pt_to_jax, out_jax, atol=1e-5)
