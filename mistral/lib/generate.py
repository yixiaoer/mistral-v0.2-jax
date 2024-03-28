from typing import Callable, NamedTuple

import einops as op
from jax import Array, lax, vmap
from jax.nn import softmax
import jax.numpy as jnp
import jax.random as jrand
from transformers import AutoTokenizer, MistralForCausalLM

from ..model.kvcache import KVCache
from ..model.mistral_lm import MistralLMParams, forward_mistral_lm
from ..model.rotary_embedding import RotaryValues, get_rotary_values_at_position, make_rotary_values

# TODO: eliminate this
n_heads_kv = 8# model.config.num_key_value_heads
n_hidden_layers = 32 ## model.config.num_hidden_layers
k_v_dimension = 128
k_v_cache_n = 2

def generate(params: MistralLMParams, tokenizer: AutoTokenizer, sentences: list[str], max_length: int, max_new_tokens: int, *, key: Array | None = None, top_k: int | None = None, top_p: float | None = None, temperature: float = 1., beam_nums: int | None = None) -> Array:
    """
    Generates text completions or continuations for a given list of input sentences using the specified language model parameters and tokenizer.

    This function utilizes various generation strategies (such as sampling and beam search) based on the provided parameters to generate text that is contextually relevant to the input sentences. The generation can be customized using parameters like `top_k`, `top_p`, and `temperature` to control the diversity and creativity of the output.

    Args:
        params (MistralLMParams): Parameters of the Mistral model to be used for text generation.
        tokenizer (AutoTokenizer): The tokenizer used for encoding input sentences and decoding output tokens.
        sentences (list[str]): A list of input sentences for which text completions are generated.
        max_length (int): The maximum length of the generated text (in tokens), including the input sentence.
        max_new_tokens (int): The maximum number of new tokens to generate, excluding the input sentence.
        key (Array | None, optional): A pseudo-random number generator (PRNG) key key for sampling. Defaults to None.
        top_k (int | None, optional): The number of highest probability vocabulary tokens to keep for top-k sampling. Defaults to None, indicating no top-k sampling.
        top_p (float | None, optional): The cumulative probability threshold for top-p (nucleus) sampling. Defaults to None, indicating no top-p sampling.
        temperature (float, optional): A scaling factor to apply to logits before sampling, affecting the distribution sharpness. Defaults to 1.0, indicating no scaling.
        beam_nums (int | None, optional): The number of beams for beam search. Defaults to None, indicating that beam search is not used.

    Returns:
        Array: An array of generated token IDS corresponding to the continuations for the sentence in this batch.

    Example:
        >>> generate(params, tokenizer, sentences, max_length, max_new_tokens, key=subkey, top_k=5, top_p=0.8, temperature=0.9)
        >>> generate(params, tokenizer, sentences, max_length, max_new_tokens, beam_nums=5)
    """
    # `max_length` and `max_new_tokens` jointly influence the maximum number of tokens generated
    inputs = tokenizer(sentences, padding=True, return_tensors='jax')
    eos_ids = tokenizer(tokenizer.eos_token, return_tensors='jax').input_ids[0, 1:]  # only eos_ids, not include bos_ids
    input_ids = inputs.input_ids.astype(jnp.uint16)
    output_ids = input_ids
    batch_size, batch_len = input_ids.shape
    attn_mask = inputs.attention_mask.astype(jnp.bool_)
    qk_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]
    kv_cache = None

    generate_tokens_n = min(*(max_length - jnp.sum(attn_mask == True, axis=1)), max_new_tokens)
    rotary_values_cur = make_rotary_values(batch_size, batch_len)
    rotary_values = make_rotary_values(batch_size, batch_len + generate_tokens_n)
    for idx in range(generate_tokens_n):
        if idx != 0:
            position = jnp.array((batch_len + idx - 1), jnp.int16)
            rotary_values_cur = get_rotary_values_at_position(rotary_values, position)

        if beam_nums:
            input_beams = [Beam(input_ids, jnp.array(1.), kv_cache)] if idx == 0 else output_beams
            ids_out, score_out, kv_cache_out = None, None, None
            output_beams = None

            if output_beams and output_beams[0].ids[:,-1] == jnp.repeat(eos_ids, batch_size):
                return output_beams[0].ids

            for input_beam in input_beams:
                input_ids_ = input_beam.ids if idx == 0 else input_beam.ids[:, -1].reshape(-1, 1)
                kv_cache = input_beam.kv_cache
                logits, kv_cache = forward_mistral_lm(params, input_ids_, qk_mask, rotary_values_cur, kv_cache)
                logits_ = logits[:, -1]
                prob_beams, ids_beams = lax.top_k(softmax(logits_), beam_nums)
                ids_out, score_out, kv_cache_out = prob_beams_n(input_beam, beam_nums, ids_out, score_out, kv_cache_out, prob_beams, ids_beams, kv_cache)
            output_beams = sort_beams(ids_out, score_out, kv_cache_out, beam_nums)
        else:
            logits, kv_cache = forward_mistral_lm(params, input_ids, qk_mask, rotary_values_cur, kv_cache)
            logits = logits[:, -1]

            if (top_k is None or top_k == 1) and (temperature == 0. or temperature == 1.) and top_p ==  None:
                next_token = greedy_search(logits)
            else:
                # first top_K, then top_p
                if top_k:
                    logits, tokens_ids = top_k_logits(logits, temperature=temperature, top_k=top_k)
                if top_p:
                    temperature, tokens_ids = (1., tokens_ids) if top_k else (temperature, None)
                    logits, tokens_ids = top_p_logits(logits, tokens_ids=tokens_ids, temperature=temperature, top_p=top_p)
                next_token = sampling(logits, tokens_ids, key)

            input_ids = next_token
            output_ids = jnp.concatenate((output_ids, next_token), axis=1)

        attn_mask = jnp.concatenate((attn_mask, jnp.ones((batch_size, 1), dtype=bool)), axis=1)
        qk_mask = attn_mask[:, None, None, None, :]
    return output_beams[0].ids if beam_nums and output_beams else output_ids

def greedy_search(logits: Array) -> Array:
    """
    Performs a greedy search to select the token with the highest logit.

    Args:
        logits (Array): An array of logits representing the model's predictions for the next token in the sequence.

    Returns:
        Array: The indices of the selected logits.
    """
    return jnp.argmax(logits, axis=-1).reshape(-1, 1)

def top_k_logits(logits: Array, *, temperature: float = 1.0, top_k: int) -> tuple[Array, Array]:
    """
    Filters the logits to keep only the top_k highest values, optionally scaling by temperature.

    Args:
        logits (Array): The logits array from a model's output.
        temperature (float, optional): A scaling factor for logits. Defaults to 1.0 (no scaling).
        top_k (int): The number of top elements to select from the logits.

    Returns:
        tuple[Array, Array]: A tuple containing two arrays:
            - The top_k logits after apply temperature.
            - The indices of top_k logits.
    """
    top_k_logits, top_k_ids = lax.top_k(logits, top_k)  # along the last axis
    return top_k_logits / temperature, top_k_ids

def top_p_logits(logits: Array, *, tokens_ids: Array | None = None, temperature: float = 1.0, top_p: float) -> tuple[Array, Array]:
    """
    Filters the logits based on the top_p cumulative probability threshold, optionally scaling by temperature.

    Args:
        logits (Array):  The logits array from a model's output.
        tokens_ids (Array | None, optional): The indices of tokens corresponding to the logits. Defaults to None.
        temperature (float, optional): A scaling factor for logits. Defaults to 1.0 (no scaling).
        top_p (float): The cumulative probability threshold for selecting top logits.

    Returns:
        tuple[Array, Array]: A tuple containing two arrays:
            - The top_k logits after apply temperature.
            - The indices of top_k logits.
    """
    if tokens_ids is None:
        tokens_ids = jnp.argsort(-logits, axis=-1)
        logits = jnp.take_along_axis(logits, tokens_ids, axis=-1)

    probs_sorted_cumsum = jnp.cumsum(softmax(logits, axis=-1), axis=-1)
    cutoff_index = jnp.sum(probs_sorted_cumsum < top_p, axis=-1).reshape(-1, 1)
    logits = jnp.where(jnp.arange(logits.shape[-1])[None, :] <= cutoff_index, logits, -jnp.inf)
    return logits / temperature, tokens_ids

def sampling(sampling_logits: Array, tokens_ids: Array, key: Array | None = None) -> Array:
    """
    Performs sampling from the given logits to select the next token in the sequence.

    Args:
        sampling_logits (Array): The logits to do sampling.
        tokens_ids (Array): The indices of tokens corresponding to the logits. 
        key (Array | None, optional): A pseudo-random number generator (PRNG) key key for sampling. Defaults to None.

    Returns:
        Array: An array containing the selected token IDS after sampling.
    """
    sampling_index = jrand.categorical(key, softmax(sampling_logits)).reshape(-1, 1)
    selected_token = jnp.take_along_axis(tokens_ids, sampling_index, axis=-1)
    return selected_token

class Beam(NamedTuple):
    """
    Represents a single beam in beam search. 

    - ids (Array): The array of token IDS in the current beam. This array represents the sequence of tokens generated so far in this beam.
    - score (Array): The cumulative score of the tokens in `ids`. This score is used to rank beams and decide which ones to keep during the beam search process.
    - kv_cache (KVCache): A KVCache object storing the past cache generated by the model.
    """
    ids: Array
    score: Array
    kv_cache: KVCache

def process_fun(input_beam: Beam, ids_beams: Array, ids_scores: Array) -> tuple[Array, Array]:
    """
    Updates an input beam with a set of new token IDs and scores, expanding the beam sequence and updating scores.

    Args:
        input_beam (Beam): The input beam to be updated.
        ids_beams (Array): An array of new candidate token IDS to be appended to the input beam's sequence. Each represents a potential next token.
        ids_scores (Array): An array of scores associated with each candidate token IDS in `ids_beams`.

    Returns:
        tuple[Array, Array]: A tuple containing two elements:
            - The first element is an array of updated token IDS, with each sequence expanded by one of the candidate token IDS.
            - The second element is an array of updated scores, with each score adjusted based on the corresponding candidate token's score.
    """
    vectorized_update_beam = vmap(lambda beam, new_id, new_score: 
        (jnp.concatenate([beam.ids, new_id], axis=-1), beam.score * new_score), 
        in_axes=(None, 1, 1), out_axes=(0, 0))
    return vectorized_update_beam(input_beam, ids_beams, ids_scores)

def prob_beams_n(input_beam: Beam, beam_nums: int, ids_out: Array | None , score_out: Array | None, kv_cache_out: Array | None, prob_beams: Array, ids_beams: Array, kv_cache: KVCache) -> tuple[Array | None, Array | None, Array | None]:
    """
    Expands one input beam into multiple output beams based on the probabilities of potential next n tokens.

    Args:
        input_beam (Beam): The current beam to expand.
        beam_nums (int): The number of beams to select for the next step of generation.
        ids_out (Array | None): An array to store the token IDS of previous selected beams. If provided, it will be updated with the new selections.
        score_out (Array | None): An array to store the scores of previous selected beams. If provided, it will be updated with the new scores.
        kv_cache_out (Array | None): An array to store the updated KVCache of the selected beams. If provided, it will be updated accordingly.
        prob_beams (Array): The probabilities of the next tokens for the input beam.
        ids_beams (Array): The token IDS corresponding to `prob_beams`.
        kv_cache (KVCache): The previous KVCache, to be updated based on the tokens added to the beams.

    Returns:
        tuple[Array | None, Array | None, Array | None]: A tuple containing the updated `ids_out`, `score_out`, and `kv_cache_out`, respectively. Each of these arrays represents the selected output beams' token IDS, scores, and KVCache.
   """
    batch_size = len(prob_beams)
    prob_beams = prob_beams.reshape(batch_size, -1, 1)
    ids_beams = ids_beams.reshape(batch_size, -1, 1)
    ids_out_beams, score_out_beams = process_fun(input_beam, ids_beams, prob_beams)

    ids_out = ids_out_beams if ids_out is None else jnp.concatenate([ids_out, ids_out_beams], axis=0)
    score_out = score_out_beams if score_out is None else jnp.concatenate([score_out, score_out_beams], axis=0)
    kv_cache_ = op.repeat(kv_cache, 'a b c d e f -> (repeat) c (a b d e f)', repeat=5)
    kv_cache_out = kv_cache_ if kv_cache_out is None else jnp.concatenate([kv_cache_out, kv_cache_], axis=0)
    return ids_out, score_out, kv_cache_out

def sort_beams(ids_out: Array | None, score_out: Array | None, kv_cache_out: Array | None, beam_nums: int) -> list[Beam]:
    """
    Sorts and trims a list of output beams to the top `beam_nums` beams based on their scores.

    After expanding each input beam into multiple output beams, this function is used to sort all generated output beams by their scores and keep only the top `beam_nums` beams. 

    Args:
        ids_out (Array | None): An array containing the token IDS for all output beams generated in the current step.
        score_out (Array | None): An array containing the scores for each output beam. These scores are used to rank the beams.
        kv_cache_out (Array | None): An array containing KVCache associated with each output beam.
        beam_nums (int): The number of top beams to retain after sorting. This parameter determines how many of the highest-scoring beams are kept for the next step of the beam search.

    Returns:
        list[Beam]: A list of output Beam.
   """
    _, batch_size, batch_len = ids_out.shape
    idx_out = jnp.argsort(- score_out, axis=0)[:beam_nums]
    score_out = jnp.take_along_axis(score_out, idx_out, axis=0)
    ids_out = jnp.take_along_axis(ids_out, idx_out, axis=0)
    kv_cache_out = jnp.take_along_axis(kv_cache_out, idx_out, axis=0)
    kv_cache_out = op.rearrange(kv_cache_out, 'g c (a b d e f) -> g a b c d e f', a=k_v_cache_n, b=n_hidden_layers, d=n_heads_kv, e=batch_len-1, f=k_v_dimension)

    output_beams = [Beam(ids_out[i], score_out[i], kv_cache_out[i]) for i in range(beam_nums)]
    return output_beams
