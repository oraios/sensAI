Welcome to sensAI!
==================

This site contains documentation for sensAI, the Python library for sensible AI.

sensAI is a high-level AI toolkit with a specific focus on **rapid
experimentation** for machine learning applications.
Through high levels of abstraction and integration,
sensAI minimises overhead whilst retaining a high degree of **flexibility**
for the implementation of custom solutions.

Some of sensAI's key benefits are:

* **A unifying interface to a wide variety of model classes across frameworks**

  Apply the same principles to a wide variety of models, whether they are
  neural networks, tree ensembles or non-parametric models &ndash; without
  losing the ability of exploiting each model's particular strengths.

  sensAI supports models based on PyTorch, scikit-learn, XGBoost and
  other libraries out of the box.
  Support for custom models can straightforwardly be established.

* **Adaptive, composable data processing pipelines**

  Modularise data pre-processing steps and features generation, representing
  the properties of features explicitly.

  * For each model, select a suitable subset of features, composing the
    the desired feature generators in order to obtain an initial
    input pipeline.

  * Transform the features into representations that are optimised for
    the model at hand.
    Some of the respective transformations can be automatically derived from
    the properties associated with features, others can be manually
    designed to exploit a model's specific capabilities (e.g. a tensor-based
    representation of complex, non-tabular data for neural networks).

  Strongly associate pipelines with models in order to avoid errors and
  gain the flexibility of supporting highly heterogeneous models within
  a single framework, bridging the gap to production along the way.

* **Fully integrated solutions for canonical tasks**

  Do away with boilerplate code by using high-level interfaces for model
  evaluation, model selection or feature selection.
  Log and track all relevant parameters as well as results along the way,
  using file-based logging or tracking frameworks such as MLflow.

* **Declarative semantics**

  Through its high level of abstraction, sensAI achieves largely
  declarative semantics: Focus on what to do rather than how to do it.

  Eschew the notion of external configuration for a single task, making
  your high-level code read like configuration instead.
  Gain the flexibility of specifying variations of your models and experiments
  with minimal code changes/extensions.


For a quick overview of sensAI's main features, please refer to the `README file on GitHub <https://github.com/opcode81/sensAI/blob/develop/README.md>`_.


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

