.. Mistral 7B v0.2 JAX documentation master file, created by
   sphinx-quickstart on Tue Mar 26 21:24:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Mistral 7B v0.2 JAX Documentation
================================================================

**Useful links**: `Source Repository <https://github.com/yixiaoer/mistral-v0.2-jax>`_ | `Issue Tracker <https://github.com/yixiaoer/mistral-v0.2-jax/issues>`_ 

Welcome to `Mistral 7B v0.2 JAX <https://github.com/yixiaoer/mistral-v0.2-jax>`_'s documentation! This project is the JAX implementation of `Mistral 7B v0.2 Base <https://twitter.com/MistralAILabs/status/1771670765521281370>`_, based on `Mistral 7B <https://arxiv.org/pdf/2310.06825.pdf>`_. Mistral 7B v0.2 JAX is developed using JAX from scratch, focuing on model architecture, tensor parallelism, and various generation methods, advancing the work of my earlier repository `mistral-jax <https://github.com/yixiaoer/mistral-jax>`_.

API
-----------------------------------------------------------------
.. toctree::
   :maxdepth: 2
   :caption: Mistral.Lib Modules

   mistral.lib.array_conversion
   mistral.lib.collate_fn
   mistral.lib.einshard
   mistral.lib.example_data
   mistral.lib.generate

.. toctree::
   :maxdepth: 2
   :caption: Mistral.Model Modules

   mistral.model.attention
   mistral.model.decoder_block
   mistral.model.decoder
   mistral.model.embedding
   mistral.model.mistral_lm
   mistral.model.mistral_model
   mistral.model.mlp_layer
   mistral.model.rms_norm
   mistral.model.rotary_embedding
