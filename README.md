# Mistral 7B v0.2 JAX

This project is the JAX implementation of [Mistral 7B v0.2 Base](https://twitter.com/MistralAILabs/status/1771670765521281370), advancing the work of my earlier repository [mistral 7B JAX](https://github.com/yixiaoer/mistral-jax/tree/main). 

It is supported by Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

*Go to [Mistral 7B v0.2 JAX Documentation Page](https://mistral-v02-jax.readthedocs.io/en/latest/).*

## Roadmap

- [x] Model architecture
- [x] Publish a Python library
- [x] 1D Model parallelism
- [x] Generation
    - [x] KV cache
    - [x] Left padding
    - [x] Top-K sampling / Top-p / Temperature
    - [x] Beam search
- [ ] Training

## Quick Installation

Simple installation from PyPI.

```sh
pip install mistral-v0.2-jax
```

## Usage

For usage of the Mistral 7B v0.2 Base JAX model, see the example below::

```python
import jax
import jax.numpy as jnp
from mistral_v0_2.model import convert_mistral_lm_params, forward_mistral_lm, make_rotary_values, shard_mistral_lm_params
from transformers import AutoTokenizer, MistralForCausalLM

model_dir = 'mistral-hf-7B-v0.2'  # convert first with 'Mistral 7B v0.2 Parameter Conversion' part in README
model = MistralForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

sentences = ['I have a cat.', 'There is a cat in my home.']
inputs = tokenizer(sentences, padding=True, return_tensors='jax')
input_ids = inputs.input_ids
batch_size, batch_len = input_ids.shape
attn_mask = inputs.attention_mask.astype(jnp.bool_)
qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]
rotary_values = make_rotary_values(batch_size, batch_len)

# load on CPU first to avoid OOM
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    params = convert_mistral_lm_params(model)
params = shard_mistral_lm_params(params)

logits, kv_cache = forward_mistral_lm(params, input_ids, qk_mask, rotary_values, None)
print(logits)
```

If you want to generate with this model, you can run it in the terminal:

```sh
python generate.py
```

## Install from Source

This project requires Python 3.12, JAX 0.4.26.

Git clone and create venv:

```sh
git clone https://github.com/yixiaoer/mistral-v0.2-jax.git
cd mistral-v0.2-jax

python3.12 -m venv venv
. venv/bin/activate
```

Install dependencies:

CPU:

```sh
pip install -U pip
pip install -U wheel
pip install "jax[cpu]"
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

CUDA 11:

```sh
pip install -U pip
pip install -U wheel
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

TPU VM:

```sh
pip install -U pip
pip install -U wheel
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
```

## Mistral 7B v0.2 Parameter Conversion

After downloading [model v0.2](https://models.mistralcdn.com/mistral-7b-v0-2/mistral-7B-v0.2.tar) and [tokenizer v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer.model), place them together in an `input_dir`, for example with name `mistral-7B-v0.2`.

Convert Mistral 7B v0.2 model weight to HuggingFace format by specifying an `output_dir` in the command, such as `mistral-hf-7B-v0.2`. (Later, use this directory as `model_dir` to access the model):

```sh
python convert_mistral_weight_to_hf.py --input_dir mistral-7B-v0.2 --model_size 7B --output_dir mistral-hf-7B-v0.2
```

The architecture of Mistral 7B v0.2 base remains largely consistent with previous versions.

```
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm()
        (post_attention_layernorm): MistralRMSNorm()
      )
    )
    (norm): MistralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

The updates include `"rope_theta"` from `10000.0` to `1000000.0` and `"sliding_window"` from `4096` to `null`:

```
MistralConfig {
  "_name_or_path": "mistral-hf-7B-v0.2",
  "architectures": [
    "MistralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mistral",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

```

## Problems Encountered

Encountered numerous challenges from the initial Mistral JAX implementation to the present. 

Click [Problems Part](https://github.com/yixiaoer/mistral-v0.2-jax/blob/main/problems.md) to see more details.
