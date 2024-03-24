import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import jax
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import einops as op
import jax.numpy as jnp
import jax.nn as nn
from jax import Array

import transformers
from transformers import AutoTokenizer, MistralForCausalLM, LlamaTokenizer, LlamaTokenizerFast
import torch
from typing import NamedTuple
import numpy as np
import math

from mistral.array_conversion import pt2jax, jax2pt_noncopy
from mistral.rotary_embedding import _make_weights, make_rotary_values, forward_rotary_embedding

print(jax.devices())

# Use the same tokenizer as Llama
# type(tokenizer): transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast

# change parameters
# n_rep_kv (the repetition number of q for k & v in one group)
n_rep_kv = 4 # see (4096 for q_proj/ 1024 for k_proj & v _proj); also R
d_k = d_v = 128 #; also K or V; 128 = 4096/ (4 * 8)
# n_heads(for multi-head attention with numpy broadcasting)
n_heads_kv = 8 # model.config to see num_key_value_heads; also H
# num of q = 8 (num of k/v) * 4(repetition number of q for k & v in one group)
d_model = 4096 #  model.config to see hidden_size; also M
batch_size = 1
seq_len = 6
