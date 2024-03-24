import os; os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import torch
from transformers import MistralForCausalLM

from mistral.model.attention import test_forward_attention
from mistral.model.embedding import test_forward_embedding
from mistral.model.mistral_lm import test_forward_mistral_lm
from mistral.model.rms_norm import test_forward_rms_norm

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1').to(device)  # if on GPU, JAX uses cuda:0

    test_forward_rms_norm(model)
    # test_forward_rotary_embedding()
    test_forward_embedding(model)
    test_forward_attention(model)
    test_forward_mistral_lm(model)

    print('✅ All tests passed!')

if __name__ == '__main__':
    main()
