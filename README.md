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

## Usage

If you want to use this Mitral JAX, you can use it like this:

```python
import jax
import jax.numpy as jnp
from mistral.model import convert_mistral_lm_params, forward_mistral_lm, make_rotary_values, shard_mistral_lm_params
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

If you want to generate with model, you can run it in the terminal:

```sh
python generate.py
```

## Install

This project requires Python 3.11, JAX 0.4.25.

Create venv:

```sh
python3.11 -m venv venv
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

## Troubleshooting

Encountered numerous challenges from the initial Mistral JAX implementation to the present. This section logs particularly peculiar or notably time-consuming issues, resolved or ongoing.

### Attention Layer Discrepancies in Model Architecture Conversion from PyTorch to JAX

#### Problem Description

During the implementation of the attention layer with JAX, the test output results diverged from the output of the original Mistral PyTorch model's attention layer.

#### Debugging Process and Solution

The original PyTorch model specified the shape of `q_proj`, `k_proj`, `v_proj`, and `o_proj` only with `in_features` and `out_features`, which correspond to `d_model`(`model.config` to see `hidden_size`), `n_rep_kv` (the repetition number of q for k & v in one group), `n_heads_kv` (seen in `model.config` for the number of key/value heads), `d_k` (dimension of k), and `d_v` (dimension of v). For enhanced clarity, decomposed `q_proj` into `Array` that represented more dimensions with `d_model`, `n_heads_kv`, `n_rep_kv`, `d_k`; similarly for `k_proj`, `v_proj`, and `o_proj`, into more dimensional shapes.

An error in the initial reasoning about the numbers assigned to `d_v` and `d_k` led to an incorrect scaled factor during the Scaled Dot-Product Attention computation (`/math.sqrt(d_k)`), which differed from the calculations in PyTorch model. After rectifying the parameter values, the attention results still varied.

Then compared the model's precision, the shapes of each `einsum` operation, and the operation of rotational embedding, line by line, between both models. Minor issues were corrected, yet the outputs still differed. Strangely, the initial and final lines of the results matched, but the middle sections were not consistent.

Suspecting the issue stemmed from the increased dimensional decomposition, which, despite matching matrix shapes, altered the correspondence of multiplied elements due to axis positioning, so experimented with various dimensional transpositions of the reshaped matrices. Ultimately, this approach aligned the results well with the original model!

### Overcoming Model Size Constraints with Tensor Parallelism 

#### Problem Description

After implementing the model architecture, the model was too large to be loaded onto a single device, given that each device had a capacity of 16 GB, there was a challenge in loading the entire model.

#### Debugging Process and Solution

To address this issue without making significant modifications to the overall structure of the model, the decision was made to implement model parallelism through Tensor Parallelism. This would allow the model to be conveniently split across different numbers of devices.

The existing parallelism-related APIs in JAX offered two main approaches for parallelism: `jax.Array` sharding with jit for automated computation partitioning, and manual sharding with `pmap()`, `shmap()`, or `xmap()`. The former was chosen for its simplicity and intuitiveness, using `jax.device_put()` to directly split according to the shape.

However, the existing APIs for defining the split shapes were not intuitive enough for convenient and general tensor parallelism. Inspired by the expressiveness of `einsum` operations, a new method, `einshard`, was developed. This method, comprising a parser and array sharding functionality, allows for more explicit partitioning and replication through syntax such as `a = einshard(a, '... -> 1 ...')`, `b = einshard(b, 'x y z w -> x2 y z4 w')`, or `c = einshard(b, 'x y z w -> x y1 z w')`.

While `einshard` worked well on CPUs and TPU v3-8, issues arose when running on TPU v4-32 due to its configuration of 4 hosts requiring cross-host partitioning, which `jax.device_put()` could not achieve. To overcome this, `jax.make_array_from_callback()` was used as a replacement.

Further refinements and modifications have been introduced into `einshard` to enhance its functionality and adaptability.

### Generation with KVCache for a Single Sentence

#### Problem Description

The challenge was to implement a generate function that leverages KVCache for faster and more efficient performance considering a single sentence as input. 

#### Debugging Process and Solution

The implementation began with processing a single sentence as input, without padding. It was necessary to track which layer the process had reached to update the corresponding cache in `KVCache`. To facilitate this, two parameters, `kvcache_cur` and `kvcache_pre`, after computing through all 32 layers in `forward_decoder()`, only `kv_cache_pre` needed to be passed back.

Upon establishing the basic structure of `KVCache`, the expected output was not achieved. The initial suspicion was a problem with the attention mechanism, leading to adjustments in the input tokens and the modification of `KVCache` in the attention layer. However, these adjustments disrupted the previously correct structure of KVCache, indicating that the attention mechanism was not the problem.

The realization then came that the issue lay with the rotary embedding's output position when inputting a new token and generating a new one. Adjusting the position encoding only resolved part of the issue. Later recognized that since tokens were being output sequentially, setting `qk_mask` to `None` — instead of using the same setting as when the entire sentence was input—resolved the problem, leading to the generation of the expected results.

### Handling Padding in Batch Sentence Generation

#### Problem Description

When generating tokens for a batch of sentences, padding was required. 

#### Debugging Process and Solution

Given that the generated tokens would be added to the right end of the sentences, using left padding was more intuitive. However, when inputting these left-padded batch sentences, the rotary embedding position differed from the situation without padding. To accommodate this, the positions corresponding to rotary values were shifted left by the length of the initial padding tokens.

Adjusting the rotary embedding alone did not yield correct results. After generating a token, we needed to shift the historical `KVCache` to ensure that newly added parts were positioned correctly. In calculating the attention score, the previous `KVCache` strategy involved appending the newly calculated k and v directly to the corresponding dimension. However, with left padding, the dimensions targeted by KVCache were fixed to a specific length, necessitating the removal of the foremost row after appending a new one to maintain this fixed length.

Despite these adjustments, the output remained incorrect. It was eventually realized that the `qk_mask`, previously set to `None` in the non-padding scenario, needed updating due to everything being shifted left. Shifting the `attn_mask` left and updating the `qk_mask` accordingly finally led to the generation of the desired results!

The approach method mentioned above was my initial strategy for handling padding in the batch. I used the `max_length` to pad all sentences. However, padding every sentence to a lengthy `max_length` from the start resulted in consistently large matrices in self-attention, leading to slow generation speeds. To address this, a subsequent modification was implemented: sentences within a batch were left-padded based on the length of the longest sentence in that batch. With this adjustment, the lengths of sentences increased token by token. While `attn_mask` and `qk_mask` still required updates, there was no need to continually shift KVCache and rotary values. Also, the input to the attention started with the initial padding length and gradually increased, instead of being always large, thereby improving the speed of generation.

### Challenges on TPU v4-32

#### Problem Description

To meet the higher memory demands for training, the model needed to be loaded on TPU v4-32. This shift presents several challenges.

#### Debugging Process and Solution

First, it was the previously mentioned issue related to Tensor Parallelism on TPU v4-32: its configuration across 4 hosts necessitated cross-host partitioning, replacing `jax.device_put()` with `jax.make_array_from_callback()`. Scripts should run on the whole 4 hosts.

Besides, the attention layer was initially partitioned based on `n_heads_kv` with a value 8 for `q_proj`, `k_proj`, `v_proj`, and `o_proj`, a strategy that worked well with TPU v3-8, which consisted of 8 devices. However, TPU v4-32 equipped with 16 devices, made the previous device-based partitioning approach impractical. The solution explored was to partition along different dimensions, specifically `d_k` and `d_v`, instead of `n_heads_kv`. This adjustment allowed for the expected results on TPU v4-32 ~~, but, curiously, it failed to produce the desired outcome on TPU v3-8. Currently, only one dimension `d_k` is being partitioned. Theoretically, this should result in only half of the attention mechanism being parallelized effectively, implying that the partitioning might not be entirely correct. However, the output aligns with the expected results at this stage. Further debugging efforts are ongoing to resolve this issue.~~. Although this partitioning resulted in slightly different outputs on the v3-8 at first, the discrepancies were minimal. Later, executing `jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)` to increase the computation precision led to identical results thereafter.

### Implementing Beam Search with KVCache

#### Problem Description

When implementing beam search, the challenge arose from needing to manage the correspondence between beams and scores, alongside different KVCache for each beam. 

#### Debugging Process and Solution

Initially, managing beam scores and KV caches was approached with an overly complex strategy. Later, it became clear that simply maintaining beam ids, scores, and `KVCache` together met the rest input requirements of `forward_mistral_lm`. However, modifying the `KVCache` for one beam unexpectedly affected others. This was surprising, given that JAX would allocate new memory addresses if the elements changed. The realization that the `KVCache` — a tuple containing k cache and v cache lists — did not allocate new addresses because of list-level operations like `pop` in Python, explained the confusion. ~~Copying the `KVCache` before making any modifications effectively preserved the uniqueness of each beam's `KVCache`.~~ Later, to better facilitate beam search with `batch_size` > 1, the structure of KVCache was revised: previously a tuple of two lists containing 32 `Array`, then transformed into a whole `Array`. This change allows for more convenient updates to the `KVCache` in batch scenarios where multiple sentences sorted based on scores. However, converting `KVCache` to a single `Array` simplified beam search but increased more slicing, axis addition, and concatenation in attention layer, significantly slowing down generation time compared to the previous `KVCache` structure. Following the release of Mistral 7B v0.2 base, minor modifications were made in the current repo to align with the v0.2 version, enabling the generation script (generate.py) to produce expected results for v0.2 base. Despite these adjustments, the generation process still faces the significant slowdown mentioned earlier. Further solutions to this issue are currently under consideration.
