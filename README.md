# Mistral 7B v0.2 JAX

This project is the JAX implementation of Mistral 7B v0.2 Base, download [here](https://models.mistralcdn.com/mistral-7b-v0-2/mistral-7B-v0.2.tar).

It is supported by Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).

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

## Install

This project requires Python 3.11, JAX 0.4.20.

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
