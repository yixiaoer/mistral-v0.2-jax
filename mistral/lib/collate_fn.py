from typing import Any

from jax import Array
import jax.numpy as jnp
from transformers import AutoTokenizer, MistralForCausalLM

# training data: seq_ids, seq_mask, labels, labels_mask
# testing data: seq_ids, seq_mask, labels
DataTrain = tuple[Array, Array, Array, Array]
DataTest = tuple[Array, Array, Array]

def raw_collate_fn(tokenizer: AutoTokenizer, max_length: int, batch: list[tuple[str, Any]]) -> DataTrain:
    # after tokenizer(sentence)，1(bos_token_id) is added at begining
    # bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id

    all_seq_ids = []
    all_seq_mask = []
    all_label_ids = []
    all_label_mask = []

    for input_, label_ in batch:
        seq_ids = tokenizer(input_, return_attention_mask=False).input_ids
        label_ids = tokenizer(str(label_), return_attention_mask=False).input_ids

        seq_len = len(seq_ids)
        # add bos, eos and padding tokens
        if seq_len < max_length:
            pad_len = max_length - seq_len
            seq_ids = seq_ids + [pad_id] * pad_len
            seq_mask = [True] * seq_len + [False] * pad_len
        else:
            seq_ids = seq_ids[:max_length]
            seq_mask = [True] * max_length

        label_len = len(label_ids)
        if label_len < max_length:
            pad_label_len = max_length - label_len
            label_ids = label_ids + [pad_id] * pad_label_len
            label_mask = [True] * label_len + [False] * pad_label_len
        else:
            label_ids =  label_ids[:max_length]
            label_mask = [True] * max_length

        all_seq_ids.append(seq_ids)
        all_seq_mask.append(seq_mask)
        all_label_ids.append(label_ids)
        all_label_mask.append(label_mask)
    seq_ids = jnp.array(all_seq_ids, dtype=jnp.uint16)
    seq_mask = jnp.array(all_seq_mask, dtype=jnp.bool_)
    labels_ids = jnp.array(all_label_ids, dtype=jnp.uint16)
    labels_mask = jnp.array(all_label_mask, dtype=jnp.bool_)
    return seq_ids, seq_mask, labels_ids, labels_mask

def test_collate_fn(tokenizer: AutoTokenizer, max_length: int, batch: list[tuple[str, Any]]) -> DataTrain:
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    all_seq_ids = []
    all_seq_mask = []
    all_label = []

    for input_, label_ in batch:
        seq_ids = tokenizer(input_, return_attention_mask=False).input_ids
        label_ids = tokenizer(label_, return_attention_mask=False).input_ids

        seq_len = len(seq_ids) + 2
        # add bos, eos and padding tokens
        if seq_len < max_length:
            pad_len = max_length - seq_len
            seq_ids = [bos_id] + seq_ids + [eos_id] + [pad_id] * pad_len
            seq_mask = [True] * seq_len + [False] * pad_len
        else:
            seq_ids = [bos_id] + seq_ids[:max_length]
            seq_mask = [True] * max_length

        all_seq_ids.append(seq_ids)
        all_seq_mask.append(seq_mask)
        all_label.append(label_)
    seq_ids = jnp.array(all_seq_ids, dtype=jnp.uint16)
    seq_mask = jnp.array(all_seq_mask, dtype=jnp.bool_)
    labels = jnp.array(all_label_ids, dtype=jnp.uint16)
    return seq_ids, seq_mask, labels
