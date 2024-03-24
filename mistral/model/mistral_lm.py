import jax
from jax import Array
import jax.numpy as jnp
from transformers import MistralForCausalLM

from .kvcache import KVCache
from .mistral_model import MistralModelParams, convert_mistral_model_params, forward_mistral_model, shard_mistral_model_params
from .rotary_embedding import RotaryValues, make_rotary_values
from ..lib.array_conversion import pt2jax
from ..lib.einshard import einshard

MistralLMParams = tuple[MistralModelParams, Array]

def convert_mistral_lm_params(model: MistralForCausalLM) -> MistralLMParams:
    model_params = convert_mistral_model_params(model.model)
    lm_head = pt2jax(model.lm_head.weight.T)
    return model_params, lm_head

def convert_back_mistral_lm_params(params: MistralLMParams) -> MistralForCausalLM:
    pass

def shard_mistral_lm_params(params: MistralLMParams) -> MistralLMParams:
    model_params, lm_head = params
    model_params = shard_mistral_model_params(model_params)
    lm_head = einshard(lm_head, '... -> 1 ...')
    return model_params, lm_head

def forward_mistral_lm(params: MistralLMParams, input_ids: Array, qk_mask: Array, rotary_values: RotaryValues, kv_cache_pre: KVCache) -> tuple[Array, KVCache]:
    model_params, lm_head = params
    outputs, kv_cache_pre = forward_mistral_model(model_params, input_ids, qk_mask, rotary_values, kv_cache_pre)
    logits = outputs @ lm_head
    return logits, kv_cache_pre

def test_forward_mistral_lm(model: MistralForCausalLM) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer.pad_token = tokenizer.eos_token
    sentences = ['I have a cat.']
    inputs = tokenizer(sentences, padding=True, return_tensors='pt')
    input_ids = inputs.input_ids
    attn_mask = inputs.attention_mask

    outputs_pt = model(input_ids, attn_mask)[0]
    outputs_pt_to_jax = pt2jax(outputs_pt)

    # load on CPU first to avoid OOM
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = convert_mistral_lm_params(model)
    params = shard_mistral_lm_params(params)

    input_ids_jax = pt2jax(input_ids)
    attn_mask_jax = pt2jax(attn_mask).astype(jnp.bool_)
    qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask_jax, attn_mask_jax))[:, None, None]

    batch_size, seq_len = input_ids_jax.shape
    rotary_values = make_rotary_values(batch_size, seq_len)

    outputs_jax, _ = forward_mistral_lm(params, input_ids_jax, qk_mask, rotary_values, None)

    outputs_pt_to_jax = jnp.where(attn_mask_jax[:, :, None], outputs_pt_to_jax, 0.)
    outputs_jax = jnp.where(attn_mask_jax[:, :, None], outputs_jax, 0.)
    assert jnp.allclose(outputs_pt_to_jax, outputs_jax, atol=1e-5)
