---
title: 'Hyperparameter Optimization'
description: "This chapter gives a basic tutorial for optimizing the hyperparameters of your model, using Optuna as an example."
author: "Makoto Hiramatsu"
type: chapter
---


<exercise id="1" title="Why hyperparameters matter">

The choice of hyperparameters often has a strong impact on the performance of a model.
Even if you use the same model, performance can drastically change depending on
the hyperparameters (e.g. learning rate, dimensionality of word embeddings) you use.
The following figure shows the performance change with different hyperparameters.
<img src="/part2/hyperparameter-optimization-with-optuna/hyperparameter_matters.jpg" alt="Why hyperparameters matter" />

A typical process of hyperparameter optimization is based on repeating a step of training/evaluating a model.
People just repeat this cycle for hours or even days to find good hyperparameters.
<img src="/part2/hyperparameter-optimization-with-optuna/what_is_hyperparameter_optimization.jpg" alt="What is hyperparameter optimization" />

</exercise>

<exercise id="2" title="Hyperparameter optimizers">

<img src="/part2/hyperparameter-optimization-with-optuna/automatic_hyperparameter_optimization.jpg" alt="Automatic Hyperparameter Optimization" />

Automatic hyperparameter optimization is an approach that automates this process.
An optimizer samples hyperparameter from the given search space,
trains a model using them and evaluates and performance.
In this chapter, we refer to this process as `trial`.
After a certain number of trials with different hyperparameters,
an optimizer reports the best performing hyperparameter combination.

A typical way to tune hyperparameters is random search.
Random search samples hyperparameter from a search space randomly.
We also note that random search samples a hyperparameter independently in each trial,
which means each trial doesn't affect the results of other trials.
Although random search often finds good hyperparameters,
using the history of previous trials can improve the search.

Sequential Model-based Optimization (SMBO) is an approach that iterates between fitting a model
and making choices which configuration to investigate in the next trial.
For example, [Tree-structured Parzen Estimator (TPE)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization)
was proposed as an example SMBO algorithm, which shows better performance than random search.

</exercise>


<exercise id="3" title="The CNN classifier to be tuned">

This tutorial works on sentiment analysis, one kind of text classification.
We use the [IMDb review dataset](https://ai.stanford.edu/~amaas/data/sentiment),
which contains 20,000 positive or negative reviews for training and 5,000 reviews for validating the performance of model.
If you haven't read [the tutorial for text classification](https://guide.allennlp.org/your-first-model#1), that may be helpful.

Below is a sample configuration of a CNN-based classifier.

```json
// imdb_baseline.jsonnet

local batch_size = 64;
local cuda_device = 0;
local num_epochs = 15;
local seed = 42;
local train_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl';
local validation_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl';

// hyperparameters
local embedding_dim = 128;
local dropout = 0.2;
local lr = 0.1;
local max_filter_size = 4;
local num_filters = 128;
local output_dim = 128;
local ngram_filter_sizes = std.range(2, max_filter_size);

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          embedding_dim: embedding_dim,
        },
      },
    },
    seq2vec_encoder: {
      type: 'cnn',
      embedding_dim: embedding_dim,
      ngram_filter_sizes: ngram_filter_sizes,
      num_filters: num_filters,
      output_dim: output_dim,
    },
    dropout: dropout,
  },
  data_loader: {
    batch_size: batch_size,
  },
  trainer: {
    cuda_device: cuda_device,
    num_epochs: num_epochs,
    optimizer: {
      lr: lr,
      type: 'sgd',
    },
    validation_metric: '+accuracy',
  },
}
```

Of course, we can train this model using AllenNLP CLI.
I ran `allennlp train imdb_baseline.jsonnet` five times with different random seeds.
As the result, the average of validation accuracy was 0.828 (±0.004).

</exercise>


<exercise id="4" title="Selecting and converting the hyperparameters">

Hyperparameters in the configuration are selected based on the standard recommendations.
Now we select some hyperparameters to be tuned.
For now, we optimize the following hyperparameters:

- `embedding_dim`
- `dropout`
- `lr`
- `max_filter_size`
- `num_filters`
- `output_dim`

First, we replace values of hyperparameters with `std.extVar` so that
we can load the values of hyperparameters from environment variables.
Remember that `std.parseInt` or `std.parseJson` are used for numerical parameters.

### Before

```json
local embedding_dim = 128;
local dropout = 0.2;
local lr = 0.1;
local max_filter_size = 4;
local num_filters = 128;
local output_dim = 128;
```

### After

```json
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));
local max_filter_size = std.parseInt(std.extVar('max_filter_size'));
local num_filters = std.parseInt(std.extVar('num_filters'));
local output_dim = std.parseInt(std.extVar('output_dim'));
```

You can view a final configuration by clicking `details` below.

<details>
<br>

`imdb_optuna.jsonnet`

```json
local batch_size = 64;
local cuda_device = 0;
local num_epochs = 15;
local seed = 42;
local train_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/train.jsonl';
local validation_data_path = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/imdb/dev.jsonl';

local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));
local max_filter_size = std.parseInt(std.extVar('max_filter_size'));
local num_filters = std.parseInt(std.extVar('num_filters'));
local output_dim = std.parseInt(std.extVar('output_dim'));
local ngram_filter_sizes = std.range(2, max_filter_size);

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  dataset_reader: {
    type: 'text_classification_json',
    tokenizer: {
      type: 'whitespace',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: train_data_path,
  validation_data_path: validation_data_path,
  model: {
    type: 'basic_classifier',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          embedding_dim: embedding_dim,
        },
      },
    },
    seq2vec_encoder: {
      type: 'cnn',
      embedding_dim: embedding_dim,
      ngram_filter_sizes: ngram_filter_sizes,
      num_filters: num_filters,
      output_dim: output_dim,
    },
    dropout: dropout,
  },
  data_loader: {
    batch_size: batch_size,
  },
  trainer: {
    cuda_device: cuda_device,
    num_epochs: num_epochs,
    optimizer: {
      lr: lr,
      type: 'sgd',
    },
    validation_metric: '+accuracy',
  },
}
```

</details>
<br>

Note that you can also conduct experiments by specifying environment variables.

```
embedding_dim=128 dropout=0.2 lr=0.1 \
  max_filter_size=4 num_filters=128 output_dim=128 \
  allennlp train classifier.jsonnet -s result
```

</exercise>


<exercise id="5" title="Running hyperparameter optimization with Optuna">

[Optuna](https://optuna.org) is a library, which allows users to optimize hyperparameters automatically.
Optuna provides sophisticated algorithms for searching hyperparameters, such as
TPE mentioned in the previous step, and [CMA Evolution Strategy](https://arxiv.org/abs/1604.00772)
, as well as algorithms for pruning unpromising trials such as [Hyperband](http://jmlr.org/papers/v18/16-558.html).
Before going to the next exercise, please run `pip install optuna` if you haven't installed Optuna yet.

Optuna offers an integration for AllenNLP,
named [AllenNLPExecutor](https://optuna.readthedocs.io/en/stable/reference/integration.html#optuna.integration.AllenNLPExecutor).
We can use `AllenNLPExecutor` by following steps: `Telling Optuna's hyperparameters` and `Defining search space`.

To use Optuna, we define the hyperparameter search spaces.
In Optuna, a search space is defined by creating an `objective function`.
Each hyperparameter search space is declared with `suggest_int` or `suggest_float`.
For categorical hyperparameters, you can use `suggest_categorical`.
Please see [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial)
for more information.

These `suggest` functions require two kinds of arguments at least.
The first one is the name of the hyperparameter, and the second one is the range of the values.
Note that the names of the hyperparameters should be the same as those defined in the configuration earlier.
A typical objective function looks like the following:

```python
import optuna


def objective(trial: optuna.Trial) -> float:
    trial.suggest_int("embedding_dim", 32, 256)
    trial.suggest_int("max_filter_size", 2, 6)
    trial.suggest_int("num_filters", 32, 256)
    trial.suggest_int("output_dim", 32, 256)
    trial.suggest_float("dropout", 0.0, 0.8)
    trial.suggest_float("lr", 5e-3, 5e-1, log=True)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file="./config/imdb_optuna.jsonnet",  # path to jsonnet
        serialization_dir=f"./result/optuna/{trial.number}",
        metrics="best_validation_accuracy"
    )
    return executor.run()
```

After defining search spaces using `trial.suggest_int` and `trial.suggest_float`, the `trial` object should be passed to `AllenNLPExecutor`.
The `trial` object holds all suggested values after defining search spaces using `trial.suggest_int`
and `trial.suggest_float`,  and it should be pass to `AllenNLPExecutor`.
`AllenNLPExecutor` takes four required arguments; `trial` (Optuna's object), `config_file` (path to a model configuration),
`serialization_dir` (directory for saving model snapshot, log, etc.), and `metrics` you want to optimize.
In the above example, we create an instance of `AllenNLPExecutor` as `executor`.
Once the `executor` instance is created, training is started with `executor.run()`.
In each trial step in optimization, the objective function is called and does the following steps:

1. Train a model (`executor.run()`)
2. Return a target metric on validation data (`executor.run()` returns the specified metric)

Once we've written an objective function, we can write a script for launching optimization.
In Optuna, we create a study object and pass the objective function to the `optimize()` method as follows.
You can specify a number of parameters for the hyperparameter optimization process:
- a way to save a result of optimization
- a sampler for searching hyperparameters (`TPESampler` is based on Bayesian Optimization)
- direction for optimizing (maximize or minimize)
- number of jobs for distributed training
- timeout

and more.  An example launch script is below.

```python
if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///result/trial.db",  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name="optuna_allennlp",
        direction="maximize",
    )

    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=30,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )
```

You can also use [`allennlp-optuna`](https://github.com/himkt/allennlp-optuna) for hyperparameter optimization.
After installing `allennlp-optuna`, you need to define a search space for hyperparameters in JSON:

```json
[
  {
    "type": "int",
    "attributes": {
      "name": "embedding_dim",
      "low": 32,
      "high": 256
    }
  },
  {
    "type": "int",
    "attributes": {
      "name": "max_filter_size",
      "low": 2,
      "high": 6
    }
  },
  {
    "type": "int",
    "attributes": {
      "name": "num_filters",
      "low": 32,
      "high": 256
    }
  },
  {
    "type": "int",
    "attributes": {
      "name": "output_dim",
      "low": 32,
      "high": 256
    }
  },
  {
    "type": "float",
    "attributes": {
      "name": "dropout",
      "low": 0.0,
      "high": 0.8
    }
  },
  {
    "type": "float",
    "attributes": {
      "name": "lr",
      "low": 5e-3,
      "high": 5e-1,
      "log": true
    }
  }
]
```

You can then launch optimization with the following command:

```bash
allennlp tune \
    config/imdb_optuna.jsonnet \
    config/hparams.json \
    --serialization-dir result/optuna \
    --study-name allennlp-optuna_demo \
    --timeout 36000 \
    --direction maximize
```

For more information about `allennlp-optuna` (including installation instructions),
please see the [README](https://github.com/himkt/allennlp-optuna) on GitHub.


</exercise>


<exercise id="6" title="Pruning poor performing trials">

Hyperparameter optimization often takes a long time to find good hyperparameters.
If you can find and stop unpromising trials with bad hyperparameters, you can reduce the time of optimization and get good hyperparameters faster.
Stopping unpromising trials is called `pruning`.
The following illustration shows an example of pruning.
Although the final curves at the left side of the picture are not available before finishing hyperparameter optimization,
a pruner evaluates at each epoch how promising each trial will be and stops if it predicts it to be unpromising.

<img src="/part2/hyperparameter-optimization-with-optuna/illustration_of_pruning.jpg" alt="Illustration of Pruning" />

Optuna provides an `AllenNLPPruningCallback`
which allows users to prune unpromising trials with algorithms implemented in Optuna.

You can enable a pruning callback by adding `optuner_pruner` to `callbacks` in
your jsonnet configuration (inside the `trainer` parameters).
A pruner determines whether it prune a training in each epoch,
based on the `metrics` specified in initializing `AllenNLPExecutor`.

```json
   callbacks: [  // note that you have to specify `epoch_callbacks` instead if you use AllenNLP<2.0.0
    {
      "type": "optuna_pruner",
    },
  ],
```

After enabling pruning callback, you have to specify a pruner you want to use.
In Optuna, some effient algorithms such as [SuccessiveHalving](https://arxiv.org/abs/1502.07943)
and [Hyperband](http://jmlr.org/papers/v18/16-558.html) are available.
In this example, we use `Hyperband` as the pruner.
For more information about pruners of Optuna,
please visit [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/pruners.html).

```
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(),  # pruner here!
)
study.optimize(objective, n_trials=50)
```

</exercise>


<exercise id="7" title="Working with Optuna Outputs">

If `study.optimize` successfully runs, `trial.db` would be created in the directory `result`.

You can see and analyze a result by passing `study` object to various methods implemented in Optuna.
If you want to separate an analysis from optimization, you can save the `study` (e.g. RDB) and load it in another script.

```python
study = optuna.load_study(
  storage="sqlite:///result/trial.db",
  study_name="optuna_allennlp"
)
```

Let's check the results of trials with a `pandas` dataframe.

```python
study.trials_dataframe()
```

<img src="/part2/hyperparameter-optimization-with-optuna/trials_dataframe.jpg" alt="Dataframe">

Next, we show an example of visualization of a optimization history.
To plot a history of optimization, we can use `optuna.visualization.plot_optimization_history`.
We also put the validation accuracy of a baseline model as a reference.
It shows that Optuna successfully found hyperparameters to achieve better performance.
Note that this figure shows one result of optimization.
We performed optimization five times with different random seeds and got an average validation accuracy of 0.909 (±0.002),
which outperforms the baseline by a large margin.

```python
optuna.visualization.plot_optimization_history(study)
```

<br>
<img src="/part2/hyperparameter-optimization-with-optuna/optimization_history.jpg" alt="Plot Optimization History" />

Optuna also lets you evaluate parameter importances based on finished trials.
There are two evaluators available: the default [fANOVA](http://proceedings.mlr.press/v32/hutter14.html)
and [MDI](https://papers.nips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-trees).
To show importances of hyperparameters, it uses `optuna.visualization.plot_param_importances`:

```python
optuna.visualization.plot_param_importances(study)
```

In this plot, we can see that `lr` is the most important hyperparameter in this experiment.

<br>
<img src="/part2/hyperparameter-optimization-with-optuna/hyperparameter_importance.jpg" alt="Plot Hyperparameter Importance" />

Additionally, you can export a configuration with optimized hyperparameters.

```python
dump_best_config("./imdb_optuna.jsonnet", "./best_config.json", study)
```

It will create a configuration named `best_config.json`.
This is helpful to retrain a model with the best hyperparameters.

</exercise>

<exercise id="8" title="For further information">

That concludes this guide on how to use Optuna for hyperparameter optimization.
Hopefully, you've learned how to define AllenNLP hyperparameter search space using Optuna,
run the trials for optimization, and then use the results with just a few lines of code.
For more details about Optuna, please see the [Optuna website](https://optuna.org/)
or [Optuna documentation](https://optuna.readthedocs.io/en/stable).

You can try the [example](https://colab.research.google.com/github/himkt/optuna-allennlp/blob/master/notebook/Optuna_AllenNLP.ipynb)
of `AllenNLPExecutor` on Google Colab.
Additionally, you can use Optuna with an AllenNLP model by writing your own Python script.
If you want to write Python script to tune hyperparameters,
please refer to the [colab example](https://colab.research.google.com/github/himkt/optuna-allennlp/blob/master/notebook/Optuna_AllenNLP_Custom_Loop.ipynb).

</exercise>
