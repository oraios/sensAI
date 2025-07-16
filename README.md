<p align="center" style="text-align:center">
  <img src="resources/sensai-logo.svg#gh-light-mode-only" style="width:400px">
  <img src="resources/sensai-logo-dark-mode.svg#gh-dark-mode-only" style="width:400px">
  <br>
  the Python library for sensible AI

  <div align="center" style="text-align:center">
  <a href="https://pypi.org/project/sensai/" style="text-decoration:none"><img src="https://img.shields.io/pypi/v/sensai.svg" alt="PyPI"></a>
  <a href="https://raw.githubusercontent.com/jambit/sensAI/master/LICENSE" style="text-decoration:none"><img alt="License" src="https://img.shields.io/pypi/l/sensai"></a>
  <a href="https://github.com/jambit/sensAI/actions/workflows/ci.yaml" style="text-decoration:none"><img src="https://github.com/jambit/sensAI/actions/workflows/ci.yaml/badge.svg" alt="Build status"></a>
  </div>
</p>


# About sensAI

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

So if you would normally use a library like scikit-learn or XGBoost on its own,
consider adding sensAI in order to
  * gain flexibility, straightforwardly supporting a greater variety of models,
  * increase the level of abstraction, cutting down on boilerplate,
  * improve logging and tracking with minimal effort.

While sensAI's main focus is on supervised and unsupervised machine learning,
it also provides functionality for discrete optimisation and a wide range
of general-purpose utilities that are frequently required in AI applications.

<hr>

<!-- generated with `markdown-toc -i README.md` -->
**Table of Contents**

<!-- toc -->

- [About sensAI](#about-sensai)
  * [Supervised Learning](#supervised-learning)
    + [Feature Generators](#feature-generators)
    + [Feature Generator Registry](#feature-generator-registry)
    + [(Model-Specific) Data Transformation](#model-specific-data-transformation)
    + [Vector Models](#vector-models)
    + [Evaluation](#evaluation)
    + [Tracking of Results](#tracking-of-results)
    + [Feature and Model Selection](#feature-and-model-selection)
    + [Peace of Mind](#peace-of-mind)
  * [Beyond Supervised Learning](#beyond-supervised-learning)
    + [Unsupervised Learning](#unsupervised-learning)
    + [Combinatorial Optimisation](#combinatorial-optimisation)
    + [Utilities, Utilities, Utilities](#utilities-utilities-utilities)
- [Documentation](#documentation)
  * [Integrating sensAI into a Project](#integrating-sensai-into-a-project)
- [Contributors](#contributors)

<!-- tocstop -->

## Supervised Learning

Many real-world tasks can be reduced to classification and regression problems, 
and sensAI specifically caters to the needs of these problems by providing a 
wide variety of concepts and abstractions that can render experimentation a 
breeze. 
We shall briefly review the most important ones in the following.

sensAI's models use pandas DataFrames to represent data points. 
Note that this does not limit the data to purely tabular data, as a field in a 
data frame can hold arbitrarily complex data. 
Yet the tabular case is, of course, a most common one.

### Feature Generators

A fundamental concept in sensAI is to introduce representations for features 
that
 * provide the logic for generating/extracting feature values from the original 
   data, decoupling externally provided input data from the data that is 
   suitable as input for various models
 * hold metadata on the generated features in order to support flexible 
   downstream transformations.

The fundamental abstraction for this is `FeatureGenerator`.
A `FeatureGenerator` takes as input a data frame and creates one or more features
from it, which a model is to take as input.  

To facilitate the definition of feature generators, sensAI provides a variety of
bases classes for that already cover the most common use cases.  
Here are some examples built on base classes provided by sensAI:

```python
from sensai.featuregen import FeatureGeneratorMapColumn, FeatureGeneratorMapColumnDict, \
    FeatureGeneratorTakeColumns

class FeatureGeneratorTemperature(FeatureGeneratorTakeColumns):
    """
    Takes the input column "temperature" without modifications, adding meta-information
    on how to normalise/scale the feature (using StandardScaler)
    """
    def __init__(self):
        super().__init__("temperature",
            normalisation_rule_template=DFTNormalisation.RuleTemplate(
                transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler()))


class FeatureGeneratorWeekday(FeatureGeneratorMapColumn):
    """
    Creates the categorical feature "weekday" (integer from 0=Monday to 6=Sunday)
    from the "timestamp" column, which is given as a pandas Timestamp object
    """
    def __init__(self):
        super().__init__(input_col_name="timestamp", feature_col_name="weekday",
            categorical_feature_names="weekday")

    def _create_value(self, timestamp: pd.Timestamp):
        return timestamp.weekday()


class FeatureGeneratorTimeOfDayCircular(FeatureGeneratorMapColumnDict):
    """
    From the "timestamp" column, creates two features "time_of_day_x" and
    "time_of_day_y", which correspond to the locations on the unit circle
    that the hour hand of a 24-hour clock would point to
    """
    def __init__(self):
        super().__init__(input_col_name="timestamp",
            normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True))

    def _create_features_dict(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        time_of_day_norm = (timestamp.hour + timestamp.minute / 60) / 24
        alpha = math.pi / 2 - time_of_day_norm * 2 * math.pi
        return dict(time_of_day_x=math.cos(alpha), time_of_day_y=math.sin(alpha))
```

:white_check_mark: **Modular features**

Feature engineering can be crucial, especially in non-deep learning applications, 
and crafting domain-specific feature generators will often be a critical task in
practice.

:information_source: With every feature being represented explicitly as a 
feature generator, 
we can flexibly make use of them in models and choose the ones we would like to 
apply for any given model.

### Feature Generator Registry

In order to simplify the definition of the set of features that a model is to 
make use of, we add feature generators to a registry, allowing us to refer to 
each feature generator by name.

```python
registry = FeatureGeneratorRegistry()
registry.register_factory(FeatureName.TEMPERATURE, FeatureGeneratorTemperature)
registry.register_factory(FeatureName.WEEKDAY, FeatureGeneratorWeekday)
registry.register_factory(FeatureName.TIME_OF_DAY_CIRC, FeatureGeneratorTimeOfDayCircular)
```

Instead of plain string names, the use of an Enum (like `FeatureName` above) can
be helpful for added auto-completion support in your IDE.

With such a registry, we can obtain any given set of features for use within
a model:

```python
feature_collector = registry.collect_features(FeatureName.TEMPERATURE, FeatureName.WEEKDAY)
features_df = feature_collector.get_multi_feature_generator().generate(df)
```

:white_check_mark: **Composable feature pipelines**

### (Model-Specific) Data Transformation

Depending on the type of model, the representation of the input data may need to
be adapted. For instance,
some models can directly process arbitarily represented categorical data, others
require an encoding. Some models can deal with arbitrary scales of numerical 
data, others work best with normalised data.

To handle this, sensAI provides the concept of a `DataFrameTransformer`
(DFT for short), which can be used to transform the data that is fed to a model 
after feature generation.
The most common transformers can conveniently be derived directly from the 
meta-data that is associated with features:

```python
feature_coll = registry.collect_features(*list(FeatureName))

dft_normalisation = feature_coll.create_dft_normalisation()
dft_one_hot_encoder = feature_coll.create_dft_one_hot_encoder() 
```

`DataFrameTransformers` serve three purposes in the context of sensAI models:
  * to transform the data prior to feature generation
  * to transform the data after feature generation
  * to transform the prediction targets (in which case the transformation
    must have an inverse)

:information_source: By using different `DataFrameTransformers`, models can 
flexibly use different feature and target representations.

:white_check_mark: **Model-specific data representations**

### Vector Models

Because sensAI models operate on data frames and every row in a data frame 
corresponds to a vector of data, the fundamental model class in sensAI is 
called `VectorModel`. (Note that, in computer science, a *vector* can hold 
arbitrary types of data.)

A `VectorModel` can be flexibly configured and provides fundamental 
functionality for the composition of model-specific data pipelines.
Here are three examples of model definitions:

```python
feature_coll = registry.collect_features(*list(FeatureName))

model_xgb = XGBRandomForestVectorRegressionModel() \
    .with_name("XGBoost") \
    .with_feature_collector(feature_coll, shared=True) \

model_linear = SkLearnLinearRegressionVectorRegressionModel() \
    .with_name("Linear") \
    .with_feature_collector(feature_coll, shared=True) \
    .with_feature_transformers(
        feature_coll.create_dft_one_hot_encoder())

model_rffn = ResidualFeedForwardNetworkVectorRegressionModel(
        hidden_dims=[10]*5, 
        cuda=False) \
    .with_name("RFFN") \
    .with_feature_collector(feature_coll, shared=True) \
    .with_feature_transformers(
        feature_coll.create_dft_one_hot_encoder(),
        feature_coll.create_dft_normalisation()) \
    .with_target_transformer(DFTSkLearnTransformer(StandardScaler()))
```

:white_check_mark: **Declarative model specifications**  
:white_check_mark: **Composable data pipelines** 

Notice that the torch-based RFFN model uses some additional transformations
that the other models can do without.

As already indicated above, sensAI comes with a variety of ready-to-use model implementations based on libraries such as scikit-learn, PyTorch and XGBoost. 
Here's a part of the class hierarchy:

```
VectorModel
├── AveragingVectorRegressionModel
├── VectorClassificationModel
│   ├── AbstractSkLearnVectorClassificationModel
│   │   ├── LightGBMVectorClassificationModel
│   │   ├── SkLearnDecisionTreeVectorClassificationModel
│   │   ├── SkLearnKNeighborsVectorClassificationModel
│   │   ├── SkLearnLogisticRegressionVectorClassificationModel
│   │   ├── SkLearnMLPVectorClassificationModel
│   │   ├── SkLearnMultinomialNBVectorClassificationModel
│   │   ├── SkLearnRandomForestVectorClassificationModel
│   │   ├── SkLearnSVCVectorClassificationModel
│   │   ├── XGBGradientBoostedVectorClassificationModel
│   │   └── XGBRandomForestVectorClassificationModel
│   ├── CategoricalNaiveBayesVectorClassificationModel
│   ├── KNearestNeighboursClassificationModel
│   └── TorchVectorClassificationModel
│       ├── LSTNetworkVectorClassificationModel
│       └── MultiLayerPerceptronVectorClassificationModel
└── VectorRegressionModel
    ├── AbstractSkLearnMultiDimVectorRegressionModel
    │   ├── SkLearnKNeighborsVectorRegressionModel
    │   ├── SkLearnLinearLassoRegressionVectorRegressionModel
    │   ├── SkLearnLinearRegressionVectorRegressionModel
    │   ├── SkLearnLinearRidgeRegressionVectorRegressionModel
    │   ├── SkLearnLinearSVRVectorRegressionModel
    │   ├── SkLearnMultiLayerPerceptronVectorRegressionModel
    │   └── SkLearnSVRVectorRegressionModel
    ├── AbstractSkLearnMultipleOneDimVectorRegressionModel
    │   ├── LightGBMVectorRegressionModel
    │   ├── SkLearnDecisionTreeVectorRegressionModel
    │   ├── SkLearnDummyVectorRegressionModel
    │   ├── SkLearnExtraTreesVectorRegressionModel
    │   ├── SkLearnGradientBoostingVectorRegressionModel
    │   ├── SkLearnRandomForestVectorRegressionModel
    │   ├── XGBGradientBoostedVectorRegressionModel
    │   └── XGBRandomForestVectorRegressionModel
    ├── KNearestNeighboursRegressionModel
    ├── KerasMultiLayerPerceptronVectorRegressionModel
    └── TorchVectorRegressionModel
        ├── MultiLayerPerceptronVectorRegressionModel
        └── ResidualFeedForwardNetworkVectorRegressionModel
```

:information_source: The implementation of custom models is straightforward.

Especially for neural network-based models, you'll usually want to define your
own model architectures. 
sensAI's base classes for torch-based models provide many high-level 
abstractions to facilitate the use of arbitrarily complex models (which
may require complex transformations of the original inputs into 
tensor-based representations). See our tutorial on neural network models.


### Evaluation

Evaluating the performance of models can be a chore. 
sensAI's high-level evaluation classes severely cut down on the boiler plate, 
allowing you to focus on what matters.

```
io_data = InputOutputData.from_data_frame(df, "target")

ev = RegressionModelEvaluation(io_data,
    RegressionEvaluatorParams(data_splitter=DataSplitterFractional(0.8)))

ev.compare_models([model_xgb, model_linear, model_rffn])
```

They can be flexibly adapted to your needs. 
You can inject evaluation metrics, mechanisms for the splitting of data,
apply cross-validation, create plots that visualize model performance,
compare model performance using multiple datasets, and much more.

:white_check_mark: **Do away with boilerplate**  
:white_check_mark: **Retain flexibility**

### Tracking of Results

sensAI supports two mechanisms for the tracking of results:
  * Writing results directly to the file system
  * Using a tracking framework such as MLflow

Here's an example where we add both to our regression experiment:

```python
sensai.util.logging.configure()

experiment_name = "MyRegressionExperiment"
run_id = datetime_tag()

# create experiment for tracking with MLflow
tracked_experiment = MLFlowExperiment(experiment_name, 
    tracking_uri="", 
    context_prefix=run_id + "_",
    add_log_to_all_contexts=True)

# create file system result writer and enable file logging
result_writer = ResultWriter(os.path.join("results", experiment_name, run_id))
sensai.util.logging.add_file_logger(result_writer.path("log.txt"))

# apply model evaluation with tracking enabled
ev.compare_models([model_xgb, model_linear, model_rffn],
    tracked_experiment=tracked_experiment,
    result_writer=result_writer)
```

:white_check_mark: **Appropriately persist results**

### Feature and Model Selection

sensAI provides convenient abstractions for feature selection, model selection
and hyperparameter optimisation.

Through its modular design, sensAI's representations can be straightforwardly 
combined with other libraries that are specialised for such purposes
(e.g. hyperopt or optuna).

### Peace of Mind

sensAI developers are dedicated to providing long-term compatibility.
In contrast to other machine learning libraries, we do our best to retain
backward compatibility of newer versions of sensAI with persisted models 
from older versions.

We use semantic versioning to indicate source-level compatibility and
will indicate breaking changes in the change log.

:white_check_mark: Backward compatibility

## Beyond Supervised Learning

### Unsupervised Learning

sensAI provides extensive support for **clustering** as well as specializations
and tools for the clustering of geographic coordinates.

It also provides a very flexible implementation of *greedy agglomerative 
clustering*, an algorithm which is very useful in practice but tends to
be overlooked.

### Combinatorial Optimisation

sensAI supports combinatorial optimisation via

 * **stochastic local search**, provding implementations of
     * simulated annealing
     * parallel tempering.

   Both algorithms support adaptive (i.e. data-driven), 
   probability-based temperature schedules, greatly facilitating 
   parametrisation in practice.
 
 * **constraint programming**, by providing utilities for formulating
   and solving optimisation problems in MiniZinc.

### Utilities, Utilities, Utilities

sensAI's `util` package contains a wide range of general utilities, including
 * caching mechanisms (using SQLite, pickle and MySQL as backends)
 * string conversion utilities (the `ToStringMixin` is incredibly flexible)
 * data structures (e.g. for tree-map-style lookups)
 * logging and profiling utilities
 * I/O utilities
 * multi-processing tools (e.g. a debugger for pickle errors)
 * etc.

# Documentation

 * [Reference documentation and tutorials](https://opcode81.github.io/sensAI/docs/)

   At this point, the documentation is still limited, but we plan to add 
   further tutorials and overview documentation in the future.
  
   For all the things we do not yet cover extensively, we encourage you to use 
   your IDE to browse class hierarchies and discover functionality by using 
   auto-completion.

   If you have a usage question, don't hesitate to add an issue on GitHub.

 * [Beyond Jupyter: A Refactoring Journey](https://github.com/aai-institute/beyond-jupyter-spotify-popularity)

   Explore this lecture series on software design in machine learning, in
   which sensAI is prominently featured.
   Our *Refactoring Journey* shows how a use case that is
   initially implemented as a Jupyter notebook can be successively refactored
   in order to improve the software design, gain flexibility for experimentation,
   and ultimately arrive at a solution that could directly be deployed for
   production.



## Integrating sensAI into a Project

sensAI can be integrated into your project in several ways: 

1. **Install it as a library** with `pip install sensai`.
   Choose this option as a regular user of sensAI with no intention of extending
   the library as part of your work.
2. **Include sensAI's source code as a package within your project** (e.g. in `src/sensai`), which you synchronise with a sensAI branch.
   Choose this option if you intend to make changes to sensAI as you develop your project. When using this option, you (and others) may even make changes to sensAI in several branches of your project (and even several projects) at the same time.
   See developer documentation in [README-dev.md](README-dev.md) for details on how synchronisation works.


# Contributors

<div align="center" style="text-align:center; padding:100px">
  <br>
  <a href="https://www.appliedai-institute.de" style="text-decoration:none"><img style="height:50px" src="resources/aai-institute-logo.svg" alt="appliedAI Institute"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="http://www.jambit.com" style="text-decoration:none"><img style="height:50px;" src="resources/jambit-logo.svg" alt="jambit"></a>
  <br><br>
</div>

sensAI is being developed by members of <a href="http://transferlab.ai">TransferLab</a> at 
<a href="https://www.appliedai-institute.de">appliedAI Institute for Europe gGmbh</a>.  
The library was originally created by the machine intelligence group at [jambit GmbH](http://www.jambit.com) and was applied in many research and pre-development projects.

The main contributors are <a href="https://github.com/opcode81">Dominik Jain</a>, <a href="https://github.com/MischaPanch">Michael Panchenko</a>, and <a href="https://github.com/schroedk">Kristof Schröder</a>.

External contributions are welcome.
