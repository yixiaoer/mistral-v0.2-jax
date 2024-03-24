import jax

from functools import partial

from jax import Array, value_and_grad
import jax.numpy as jnp
from jax_smi import initialise_tracking
import optax
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MistralForCausalLM

from mistral.lib.example_data import ExampleDataset
from mistral.lib.collate_fn import raw_collate_fn
from mistral.model.mistral_lm import MistralLMParams, forward_mistral_lm
from mistral.model.rotary_embedding import RotaryValues, make_rotary_values
from mistral.model.mistral_lm import convert_mistral_lm_params, shard_mistral_lm_params

initialise_tracking()

@value_and_grad
def loss_and_grad(params: MistralLMParams, seq_ids: Array, qk_mask: Array, label_ids: Array, label_mask: Array, rotary_values: RotaryValues) -> Array:
    logits, _ = forward_mistral_lm(params, seq_ids, qk_mask, rotary_values, None)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=label_ids)
    loss = jnp.mean(loss, where=label_mask)
    return loss

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

    epochs = 3
    learning_rate = 1e-5
    batch_size = 2
    max_length = 32
    rotary_values = make_rotary_values(batch_size, max_length)

    train_data = ExampleDataset('train')
    # test_data = ExampleDataset('test')
    collate_fn = partial(raw_collate_fn, tokenizer, max_length)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    for _ in range(epochs):
        for step, (seq_ids, seq_mask, label_ids, label_mask) in enumerate(train_dataloader):
            qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', seq_mask, seq_mask))[:, None, None]
            loss, grad = loss_and_grad(params, seq_ids, qk_mask, label_ids, label_mask, rotary_values)

            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            jax.debug.callback(print, loss)

if __name__ == '__main__':
    main()
