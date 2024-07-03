import os; os.environ['JAX_PLATFORMS'] = 'cpu'

import pickle

from transformers import MistralForCausalLM

from mistral_v0_2.model.mistral_lm import convert_mistral_lm_params

model_dir = 'mistral-hf-7B-v0.2'  # convert first with 'Mistral 7B v0.2 Parameter Conversion' part in README
model = MistralForCausalLM.from_pretrained(model_dir)
params = convert_mistral_lm_params(model)

with open('/dev/shm/model.pickle', 'wb') as f:
    pickle.dump(params, f)
