---
title: 'Hyperparameter Optimization with Optuna'
description: "This chapter gives a basic tutorial for optimizing the hyperparameters of your model."
type: chapter
---

<textblock>

In this chapter, we'll give a quick tutorial of hyperparameter optimization for your AllenNLP model.

</textblock>

<exercise id="1" title="Hyperparameters matter!">

The choice of hyperparameters often has a strong impact to the performance of a model.
Even if you use the same model, performance can drastically change depending on the hyperparameters (e.g. learning rate, dimensionality of word embeddings) you use.
Following figure shows the performance change with different hyperparameters.
<img src="/part3/hyperparameter-optimization-with-optuna/hyperparameter_matters.jpg" alt="Hyperparameters matter!" />

A typical process of hyperparameter optimization is based on repeating a step of training/evaluating a model.
People just repeat this cycle for every hours or even days for finding good hyperparameters.
<img src="/part3/hyperparameter-optimization-with-optuna/what_is_hyperparameter_optimization.jpg" alt="What is hyperparameter optimization" />

<a href="https://optuna.org">Optuna</a> is a library, which allows users to optimize hyperparameters automatically.
Optuna provides sophisticated algorithms for searching hyperparameters, such as [Tree-structured Parzen Estimator](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization) and
[CMA Evolution Strategy](https://arxiv.org/abs/1604.00772), as well as algorithms for pruning unpromising trials, such as [Hyperband](http://jmlr.org/papers/v18/16-558.html).

Before going to the next exercise, please run `pip install optuna` if you haven't installed Optuna yet.

</exercise>

<exercise id="2" title="Building your model">

This tutorial works on sentiment analysis, one kind of text classification.
We use the [IMDb review dataset](https://ai.stanford.edu/~amaas/data/sentiment), which contains 20,000 positive or negative reviews for training and 5,000 reviews for validating the performance of model.
If you haven't read <a href="https://guide.allennlp.org/your-first-model#1">the tutorial for text classification</a>, that may be helpful.

Below is a sample configuration of a CNN-based classifier.
Note that this model has six hyperparameters: `embedding_dim`, `dropout`, `lr`, `max_filter_size`, `num_filters`, and `output_dim`.

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

These hyperparameters are selected based on the standard recommendations.
Of course, we can train this model using AllenNLP CLI.
I ran `allennlp train imdb_baseline.jsonnet` five times with different random seeds.
As the result, the average of validation accuracy was 0.828 (±0.004).

Let's dive into hyperparameter optimization.
Optuna offers a integration for AllenNLP, named <a href="https://optuna.readthedocs.io/en/stable/reference/integration.html#optuna.integration.AllenNLPExecutor">`AllenNLPExecutor`</a>.
We can use `AllenNLPExecutor` by following steps: `Telling Optuna's hyperparameters` and `Defining search space`.

## I: Telling Optuna’s Hyperparameters

First, we replace values of hyperparameters with `std.extVar` to tell Optuna what parameters to be optimized.
Remember that `std.parseInt` or `std.parseJson` are used for numerical parameters.

### Before

```json
local embedding_dim = 128;
local dropout = 0.2;
local lr = 0.1;
local max_filter_size = 4;
local num_filters = 128;
local output_dim = 128;
local ngram_filter_sizes = std.range(2, max_filter_size);
```

### After

```json
local embedding_dim = std.parseInt(std.extVar('embedding_dim'));
local dropout = std.parseJson(std.extVar('dropout'));
local lr = std.parseJson(std.extVar('lr'));
local max_filter_size = std.parseInt(std.extVar('max_filter_size'));
local num_filters = std.parseInt(std.extVar('num_filters'));
local output_dim = std.parseInt(std.extVar('output_dim'));
local ngram_filter_sizes = std.range(2, max_filter_size);
```

That's it. You can view a final configuration by clicking `details` below.

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

## II: Defining Search Space for Hyperparameters in Pythonic way

Now that you have created the config, the next step is defining the hyperparameter search spaces.
In Optuna, a search space is defined by creating an `objective function`.
Each hyperparameter search space is declared with `suggest_int` or `suggest_float`.
For categorical hyperparameters, you can use `suggest_categorical`.
Please see <a href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial">Optuna documentation</a> for more information.

These suggest functions require two kinds of arguments at least.
The first one is the name of hyperparameter, and the second one is the range of the values.
Note that the names of the hyperparameters should be the same as those defined in the configuration earlier.
A typical objective function looks like following:

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
The `trial` object holds all suggested values after defining search spaces using `trial.suggest_int` and `trial.suggest_float`,  and it should be pass to `AllenNLPExecutor`.
`AllenNLPExecutor` takes four required arguments; `trial` (Optuna's object), `config_file` (path to a model configuration),
`serialization_dir` (directory for saving model snapshot, log, etc.), and `metrics` you want to optimize.
In the above example, we create an instance of `AllenNLPExecutor` as `executor`.
Once the `executor` instance is created, training is started with `executor.run()`.
In each trial step in optimization, the objective function is called and does the following steps:

1. Train a model (`executor.run()`)
2. Return a target metric on validation data (`executor.run()` returns the specified metric)

Now, we finished defining objective function.
Let's write Optuna stuff for launching optimization!
In Optuna, we create a study object and pass the objective function to the `optimize()` method as follows.
You can specify something; a way to save a result of optimization, a sampler for searching hyperparameters (`TPESampler` is based on Bayesian Optimization),
direction for optimizing (maximize or minimize), number of jobs for distributed training, or timeout.

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

</exercise>

<exercise id="3" title="Results of Hyperparameter Optimization">

If `study.optimize` successfully runs, `trial.db` would be created in the directory `result`.

[Tip] You can change a names of database file and directory by changing the value of `storage` in `optuna.create_study`.

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

<img src="/part3/hyperparameter-optimization-with-optuna/trials_dataframe.jpg" alt="Dataframe">

Next, visualize a history of optimization.
To plot a history of optimization, we can use `optuna.visualization.plot_optimization_history`.
I also put a validation accuracy of baseline model as a reference.
It shows that Optuna successfully found hyperparameters to achieve better performance.
Note that this figure shows one result of optimization.
I performed optimization five times with different random seeds and got the average validation accuracy of 0.909 (±0.002), which outperforms the baseline by a large margin.

```python
optuna.visualization.plot_optimization_history(study)
```

<br>
<img src="/part3/hyperparameter-optimization-with-optuna/optimization_history.jpg" alt="Plot Optimization History" />

From the recent release of Optuna, we can evaluate parameter importances based on finished trials.
There are two evaluators available: the default [fANOVA](http://proceedings.mlr.press/v32/hutter14.html) and [MDI](https://papers.nips.cc/paper/4928-understanding-variable-importances-in-forests-of-randomized-trees).
To show importances of hyperparameters, it uses `optuna.visualization.plot_param_importances`:

```python
optuna.visualization.plot_param_importances(study)
```

In this plot, we can see that `lr` is the most important hyperparameter in this experiment.

<br>
<img src="/part3/hyperparameter-optimization-with-optuna/hyperparameter_importance.jpg" alt="Plot Hyperparameter Importance" />

Additionally, you can export a configuration with optimized hyperparameters.

```python
dump_best_config("./imdb_optuna.jsonnet", "./best_config.json", study)
```

It will create a configuration named `best_config.json`.
This is helpful to retrain a model with the best hyperparameters.

</exercise>

<exercise id="4" title="Writing your own script with Optuna and Use Pruners">

Additionally, you can use Optuna when you define the AllenNLP model by writing your script.
A python code of training a model with AllenNLP typically consists of main three parts: preparing data, building model, and creating trainer.
In this section, we show the example of AllenNLP+Optuna with writing Python script.
<!-- We put the python code to use AllenNLP+Optuna at the bottom of this exercise. -->

## Pruning unpromising trials

Hyperparameter optimization often takes a long time to find good hyperparameters.
If you can find and stop unpromising trials with bad hyperparameters, you can reduce the time of optimization and get good hyperparameters faster.
Stopping unpromising trials is so-called `pruning`.
The following illustration shows an example of pruning.
Although the final curves at the left side of the picture are not available before finishing hyperparameter optimization,
a pruner evaluates at each epoch how promising each trial will be and stops if it predicts to be unpromising.

<img src="/part3/hyperparameter-optimization-with-optuna/illustration_of_pruning.jpg" alt="Illustration of Pruning" />

You can use `AllenNLPPruningCallback` that is the new feature of Optuna, which allows users to prune unpromising trials with algorithms implemented in Optuna.
`AllenNLPPruningCallback` is the interface to provide the way to use these pruning algorithms in `GradientDescentTrainer`, which is the standard way to train a model in AllenNLP.
To enable a pruning, you must create the AllenNLP callback first. `AllenNLPPruningCallback` needs two arguments: a trial and a target metric.
In this example, we use `validation_accuracy` for the metric to determine if a trial should be pruned or not.
Note that `patience` is set to `None` since pruning and built-in early stopping could conflict.
The objective function looks like following:

```python
    def objective(trial):
        trainer = GradientDescentTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            validation_data_loader=validation_data_loader,
            validation_metric="+accuracy",
            patience=None,
            num_epochs=50,
            cuda_device=CUDA_DEVICE,
            serialization_dir="result",
            epoch_callbacks=[AllenNLPPruningCallback(trial, "validation_accuracy")],
        )
```

Next, you specify a pruner you want to use.
In Optuna, some effient algorithms such as [SuccessiveHaving](https://arxiv.org/abs/1502.07943) and [Hyperband](http://jmlr.org/papers/v18/16-558.html) are available.
In this example, we use `Hyperband` as the pruner.
For more information about pruners of Optuna, please visit <a href="https://optuna.readthedocs.io/en/stable/reference/pruners.html">Optuna documentation</a>.

```
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(),  # pruner here!
)
study.optimize(objective, n_trials=50)
```

That's it. Now you can use Optuna in your AllenNLP code!
The following script is the full example of the script.
<!-- [TODO: Installing Optuna is needed for executing this script.] -->
<codeblock source="part3/optuna/source" setup="part3/optuna/setup"></codeblock>

</exercise>

<exercise id="5" title="Summary">

That concludes this guide on how to use Optuna for hyperparameter optimization. Hopefully, you've learned how to define AllenNLP hyperparameter search space using Optuna, run the trials for optimization, and then use the results with just a few lines of code.
For more details about Optuna, please see the <a href="https://optuna.org/">Optuna website</a> or <a href="https://optuna.readthedocs.io/en/stable/">Optuna documentation</a>.

</exercise>
