import os; os.environ['JAX_PLATFORMS'] = 'cpu'

import pickle

from transformers import MistralForCausalLM

from mistral.mistral_lm import convert_mistral_lm_params

model = MistralForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
params = convert_mistral_lm_params(model)

with open('params.pickle', 'wb') as f:
    pickle.dump(params, f)
