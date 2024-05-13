.. Mistral 7B v0.2 JAX documentation master file, created by
   sphinx-quickstart on Tue Mar 26 21:24:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Mistral 7B v0.2 JAX Documentation
================================================================

**Useful links**: `Source Repository <https://github.com/yixiaoer/mistral-v0.2-jax>`_ | `Issue Tracker <https://github.com/yixiaoer/mistral-v0.2-jax/issues>`_ 

Welcome to `Mistral 7B v0.2 JAX <https://github.com/yixiaoer/mistral-v0.2-jax>`_'s documentation! This project is the JAX implementation of `Mistral 7B v0.2 Base <https://twitter.com/MistralAILabs/status/1771670765521281370>`_, based on `Mistral 7B <https://arxiv.org/pdf/2310.06825.pdf>`_. Mistral 7B v0.2 JAX is developed using JAX from scratch, focuing on model architecture, tensor parallelism, and various generation methods, advancing the work of my earlier repository `mistral-jax <https://github.com/yixiaoer/mistral-jax>`_.

Installation
-----------------------------------------------------------------

Simple installation from `PyPI <https://pypi.org/project/mistral-v0.2-jax/>`_ using::

   pip install mistral-v0.2-jax

You may also install directly from GitHub by visiting `Install From Source <https://github.com/yixiaoer/mistral-v0.2-jax?tab=readme-ov-file#install-from-source>`_ and following the instructions that are appropriate for your device.

API
-----------------------------------------------------------------
For a full idex, see the :ref:`genindex`.

.. toctree::
   :maxdepth: 2
   :caption: Mistral_v0_2.Lib Modules

   mistral_v0_2.lib.array_conversion
   mistral_v0_2.lib.collate_fn
   mistral_v0_2.lib.example_data
   mistral_v0_2.lib.generate

.. toctree::
   :maxdepth: 2
   :caption: Mistral_v0_2.Model Modules

   mistral_v0_2.model.attention
   mistral_v0_2.model.decoder_block
   mistral_v0_2.model.decoder
   mistral_v0_2.model.embedding
   mistral_v0_2.model.mistral_lm
   mistral_v0_2.model.mistral_model
   mistral_v0_2.model.mlp_layer
   mistral_v0_2.model.rms_norm
   mistral_v0_2.model.rotary_embedding

License
-----------------------------------------------------------------

Mistral 7B v0.2 JAX  is licensed under the `MIT License <https://github.com/yixiaoer/mistral-v0.2-jax/blob/main/LICENSE>`_.
