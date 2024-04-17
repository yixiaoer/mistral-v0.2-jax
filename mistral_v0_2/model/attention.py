import math

import einops as op
from einshard import einshard
import jax
from jax import Array
import jax.numpy as jnp
import torch
from transformers import MistralForCausalLM
from transformers.models.mistral.modeling_mistral import MistralAttention

from .kvcache import KVCache
from .rotary_embedding import RotaryValues, make_rotary_values, forward_rotary_embedding
from ..lib.array_conversion import pt2jax

# TODO: eliminate this
d_model = 4096
n_rep_kv = 4
n_heads_kv = 8
d_k = d_v = 128

AttentionParams = tuple[Array, Array, Array, Array]

def convert_attention_params(self_attn: MistralAttention) -> AttentionParams:
    """
    Converts the attention parameters from MistralAttention HuggingFace format with PyTorch `tensor` to `jax.Array`.

    Args:
        self_attn (MistralAttention): The attention parameters in MistralAttention.

    Returns:
        AttentionParams: The attention parameters converted into the AttentionParams with JAX.
    """
    q_proj = self_attn.q_proj.weight.data
    k_proj = self_attn.k_proj.weight.data
    v_proj = self_attn.v_proj.weight.data
    o_proj = self_attn.o_proj.weight.data

    q_proj_jax = pt2jax(q_proj.T).reshape(d_model, n_heads_kv, n_rep_kv, d_k).transpose(0, 2, 1, 3)
    k_proj_jax = pt2jax(k_proj.T).reshape(d_model, n_heads_kv, d_k)
    v_proj_jax = pt2jax(v_proj.T).reshape(d_model, n_heads_kv, d_v)
    o_proj_jax = pt2jax(o_proj.T).reshape(n_heads_kv, n_rep_kv, d_v, d_model).transpose(1, 0, 2, 3)

    return q_proj_jax, k_proj_jax, v_proj_jax, o_proj_jax

def convert_back_attention_params():
    pass

def shard_attention_params(params: AttentionParams) -> AttentionParams:
    """
    Shard the attention parameters for distributed computing.

    Args:
        params (AttentionParams): The attention parameters.

    Returns:
        AttentionParams: The attention parameters modified with tensor parallelism, allowing for distributed computation across multiple devices.
    """
    q_proj, k_proj, v_proj, o_proj = params
    # q_proj = einshard(q_proj, 'm r h k -> m r h* k')
    # k_proj = einshard(k_proj, 'm h k -> m h* k')
    # v_proj = einshard(v_proj, 'm h v -> m h* v')
    # o_proj = einshard(o_proj, 'r h v m -> r h* v m')
    q_proj = einshard(q_proj, 'm r h k -> m r h k*')
    k_proj = einshard(k_proj, 'm h k -> m h k*')
    v_proj = einshard(v_proj, 'm h v -> m h v*')
    o_proj = einshard(o_proj, 'r h v m -> r h v* m')
    return q_proj, k_proj, v_proj, o_proj

def forward_attention(params: AttentionParams, seq: Array, qk_mask: Array, rotary_values: RotaryValues, kv_cache_cur: KVCache, kv_cache_pre: KVCache) -> tuple[Array, KVCache, KVCache]:
    """
    Performs the forward pass of the attention mechanism using.

    This function executes the attention mechanism on the input sequence `seq` using the provided attention parameters. 

    Args:
        params (AttentionParams): The attention parameters.
        seq (Array): The input sequences on which attention is to be applied.
        qk_mask (Array): The qk mask for the attention mechanism, determining which parts of the sequence are allowed to attend to each other.
        rotary_values (RotaryValues): Rotary positional embeddings values.
        kv_cache_cur (KVCache): The current KVCache.
        kv_cache_pre (KVCache): The previous KVCache.

    Returns:
        tuple[Array, KVCache, KVCache]: A tuple containing the output sequence after applying attention, and the updated current and previous KVCache.
    """
    q_proj_jax, k_proj_jax, v_proj_jax, o_proj_jax = params

    # for q, the seq is src_seq, 
    # for k and v, the seq is des_seq,
    # in self_atten the src_ and des_seq are the same

    # q.shape: (1 batch_size, 4 n_rep_kv, 8 n_head, 6 seq_len ?, 128 k_dimension)
    # k.shape: (1 batch_size, 8 n_head, 6 seq_len, 128 k_dimension)
    # v.shape: (1 batch_size, 8 n_head, 6 seq_len, 128 v_dimension)

    # einsum can use to apply matrix multiplication
    q = op.einsum(seq, q_proj_jax, 'b s m, m r h k -> b r h s k')
    k = op.einsum(seq, k_proj_jax, 'b d m, m h k -> b h d k')
    v = op.einsum(seq, v_proj_jax, 'b d m, m h v -> b h d v')

    # before self attention, add position information
    # q.shape: (1 batch_size, 4, 8, 6 seq_len, 128)
    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)

    # KVCache to optimize generation
    if kv_cache_pre is not None:
        layer_n = 0 if kv_cache_cur is None else kv_cache_cur.shape[1]
        k = jnp.concatenate((kv_cache_pre[0, layer_n, ...], k), axis=-2)
        v = jnp.concatenate((kv_cache_pre[1, layer_n, ...], v), axis=-2)

    k_cache_cur = k[None, ...] if kv_cache_cur is None else jnp.concatenate((kv_cache_cur[0, ...], k[None, ...]), axis=0)
    v_cache_cur = v[None, ...] if kv_cache_cur is None else jnp.concatenate((kv_cache_cur[1, ...], v[None, ...]), axis=0)
    kv_cache_cur = jnp.concatenate((k_cache_cur[None, ...], v_cache_cur[None, ...]), axis=0)
    # self-attention
    # (1 batch_size, 4 repetition, 8 head number, 6 seq_len, 6 seq_len)
    # Scaled Dot-Product Attention as 3.2.1 equation(1) in orginal Transformer paper
    qk = jnp.einsum('brhsk,bhdk->brhsd', q, k) / math.sqrt(d_k)

    qk = jax.nn.softmax(qk, where=qk_mask, initial=0.)
    qkv = jnp.einsum('brhsd,bhdv->brhsv', qk, v)
    out = jnp.einsum('brhsv,rhvm->bsm', qkv, o_proj_jax)
    return out, kv_cache_cur, kv_cache_pre

def test_forward_attention(model: MistralForCausalLM) -> None:
    """
    Tests the forward attention mechanism.

    This function is designed to validate the functionality and correctness of the attention mechanism with JAX.

    Args:
        model (MistralForCausalLM): PyTorch Mistral model to compare with this implementation.

    Returns:
        None.
    """
    batch_size = 1
    seq_len = 6

    self_attn_pt = model.model.layers[0].self_attn
    seq_pt = torch.rand(batch_size, seq_len, d_model, device=model.device)
    attention_mask_pt = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=model.device))
    attention_mask_pt_ = torch.where(attention_mask_pt, 0., -torch.inf)

    out_pt = self_attn_pt(seq_pt, attention_mask=attention_mask_pt_)[0]

    params = convert_attention_params(self_attn_pt)

    seq_jax = pt2jax(seq_pt)
    attention_mask_jax = pt2jax(attention_mask_pt)
    rotary_values = make_rotary_values(batch_size, seq_len)
    out_jax, _, _ = forward_attention(params, seq_jax, attention_mask_jax, rotary_values, None, None)

    assert jnp.allclose(out_jax, pt2jax(out_pt), atol=1e-5)
