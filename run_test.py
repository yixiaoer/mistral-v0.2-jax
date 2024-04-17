import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

from transformers import MistralForCausalLM

from mistral_v0_2.model.attention import test_forward_attention
from mistral_v0_2.model.embedding import test_forward_embedding
from mistral_v0_2.model.mistral_lm import test_forward_mistral_lm
from mistral_v0_2.model.rms_norm import test_forward_rms_norm

def main():
    model_dir = 'mistral-hf-7B-v0.2'  # convert first with 'Mistral 7B v0.2 Parameter Conversion' part in README
    model = MistralForCausalLM.from_pretrained(model_dir)

    test_forward_rms_norm(model)
    # test_forward_rotary_embedding()
    test_forward_embedding(model)
    test_forward_attention(model)
    test_forward_mistral_lm(model)

    print('✅ All tests passed!')

if __name__ == '__main__':
    main()
