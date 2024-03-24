from typing import Callable

import jax
import jax.random as jrand
import torch
from transformers import AutoTokenizer, MistralForCausalLM

jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)


from mistral.model.mistral_lm import convert_mistral_lm_params, shard_mistral_lm_params
from mistral.lib.generate import generate

def main():
    jax.distributed.initialize()
    model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer.pad_token = tokenizer.eos_token

    if jax.local_device_count() == 8:
        # if it's V3-8, load on CPU first to avoid OOM
        cpu_device = jax.devices('cpu')[0]
        with jax.default_device(cpu_device):
            params = convert_mistral_lm_params(model)
    elif jax.local_device_count() == 4:
        # if it's V4-32
        params = convert_mistral_lm_params(model)
    params = shard_mistral_lm_params(params)

    sentences = ['How have you been?', 'The Lord of the Rings is a']
    max_new_tokens = 32
    max_length = 64
    key = jrand.key(42)
    key, subkey = jrand.split(key)

    output_ids = generate(params, tokenizer, sentences, max_length, max_new_tokens, sliding_window=model.config.sliding_window)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    output_ids_bs = generate(params, tokenizer, sentences, max_length, max_new_tokens, beam_nums=5)
    output_bs = tokenizer.batch_decode(output_ids_bs, skip_special_tokens=True)

    output_id_sampling = generate(params, tokenizer, sentences, max_length, max_new_tokens, key=subkey, top_k=5, top_p=0.8, temperature=0.9)
    output_sampling = tokenizer.batch_decode(output_id_sampling, skip_special_tokens=True)

    inputs_pt = tokenizer(sentences, padding='max_length', max_length=max_length, return_tensors='pt')
    inputs_pt_bs = tokenizer(sentences, padding='max_length', max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        generated_pt = model.generate(input_ids=inputs_pt.input_ids, attention_mask=inputs_pt.attention_mask, do_sample=False, max_new_tokens=max_new_tokens, max_length=max_length)
        generated_pt_bs = model.generate(input_ids=inputs_pt_bs.input_ids, attention_mask=inputs_pt_bs.attention_mask, do_sample=False, max_new_tokens=max_new_tokens, max_length=max_length, num_beams=5)
    output_pt = tokenizer.batch_decode(generated_pt, skip_special_tokens=True)
    output_pt_bs = tokenizer.batch_decode(generated_pt_bs, skip_special_tokens=True)

    print(f'JAX(sampling) output: {output_sampling}')
    print(f'JAX output: {output}')
    print(f'PyTorch output: {output_pt}')
    print(f'JAX output (Beam Search): {output_bs}')
    print(f'PyTorch output (Beam Search): {output_pt_bs}')
    print(f'JAX output == PyTorch output: {output == output_pt}')
    print(f'JAX output == PyTorch output (Beam Search): {output_bs == output_pt_bs}')
    # JAX(sampling) output: ["How have you been?\nWe're still working our dayj0bes, and I think it has taken me until today for it finally feel that the year is starting to roll", 'The Lord of the Rings is a trilogy of high fantasy novels written by English author J. R. R. Tolkien. The trilogy, collectively titled The Lord of']
    # JAX output: ['How have you been? I’ve been busy with work and life, but I’m still here. I’m still here.\n\nI’ve been thinking about this', 'The Lord of the Rings is a series of three epic fantasy novels written by J. R. R. Tolkien. The books tell of the quest of a group of heroes to destroy a']
    # PyTorch output: ['How have you been? I’ve been busy with work and life, but I’m still here. I’m still here.\n\nI’ve been thinking about this', 'The Lord of the Rings is a series of three epic fantasy novels written by J. R. R. Tolkien. The books tell of the quest of a group of heroes to destroy a']
    # JAX output (Beam Search): ['How have you been? It’s been a while since I’ve written a blog post, and I’ve missed it. I’ve had a lot going on in my', 'The Lord of the Rings is a series of high fantasy novels written by English author and scholar J. R. R. Tolkien. The story began as a sequel to Tolkien']
    # PyTorch output (Beam Search): ['How have you been? It’s been a while since I’ve written a blog post, and I’ve missed it. I’ve had a lot going on in my', 'The Lord of the Rings is a series of high fantasy novels written by English author and scholar J. R. R. Tolkien. The story began as a sequel to Tolkien']
    # JAX output == PyTorch output: True
    # JAX output == PyTorch output (Beam Search): True
if __name__ == '__main__':
    main()
