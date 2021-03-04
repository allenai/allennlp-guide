---
title: 'Building your model'
description: 'This chapter gives a deep dive into one of the most important components of AllenNLP—Model—and an in-depth guide to building your own model.'
type: chapter
---

<textblock>

Models play a central role in AllenNLP and NLP applications in general. They implement the
computation performed to transform input to output, and hold parameters that get optimized through
training. In this chapter, we'll give a deep dive into inner workings of AllenNLP `Models`, and
provide an in-depth guide to building, using, saving and loading your own model.

</textblock>


<exercise id="1" title="Model and Model.forward()">

AllenNLP uses the class `Model` to implement NLP models. `Model` subclasses `torch.nn.Module`,
meaning every AllenNLP `Model` is also a PyTorch `Module` and you can use it as any other PyTorch
`Modules`, e.g., calling a model (which invokes the `__call__()` method) to run the forward pass,
using `to()` to move it between devices, combining submodules to implement a larger model, and so
on.

There are, however, a few key differences between PyTorch `Modules` and AllenNLP `Models`. The most
important is the fact that `forward()` returns a dictionary, unlike most PyTorch `Modules`, which
usually return a tensor. AllenNLP `Models` also implement additional features for running and
processing model predictions, saving and loading models, getting metrics, etc., which we'll
elaborate on below.

The input/output spec of `Model.forward()` is somewhat more strictly defined than that of PyTorch
modules. Its parameters need to match field names in your [data code](/reading-data#1) exactly.
Instances created by the dataset reader are batched and converted to a set of tensors by AllenNLP
(specifically, this part happens in the [`allennlp_collate`
function](http://docs.allennlp.org/main/api/data/data_loaders/data_loader/#allennlp_collate) that the
`DataLoader` uses).  Inside our `Trainer`, batched tensors get passed to `Model.forward()` by their
original field names. The following figure shows this process:

<img src="/part2/building-your-model/forward.svg" alt="Model.forward() and its parameters" />

Note that some fields (for example, training labels) may be absent during prediction, if you want to
have your model easily made into a demo. Make sure optional `forward()` parameters have default
values as shown below, otherwise you'll get a missing argument error.

```python
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
```

In the exercise below, we create a toy model and observe how instances get batched and passed to the
`forward()` method.

<codeblock source="part2/building-your-model/model_forward" setup="part2/building-your-model/setup_model_forward"></codeblock>

As mentioned above, `Model.forward()` returns a dictionary, instead of a tensor.  Although
technically you can put anything you want in this dictionary, if you want to train your model
through backpropagation using our `Trainer`, the return value must contain a `"loss"` key pointing
to a scalar `Tensor` that represents the loss, which then gets minimized by the optimizer. For
example, here's the snippet from the forward method of `SimpleClassifier`, where a dictionary
consisting of `"probs"` and an optional `"loss"` is returned:

```python
probs = torch.nn.functional.softmax(logits)
output = {'probs': probs}
if label is not None:
    self.accuracy(logits, label)
    output['loss'] = torch.nn.functional.cross_entropy(logits, label)
return output
```

You also need to include in this dictionary any information necessary for subsequent analysis,
visualization, and so on. For example, if you want to show the predicted labels in a demo, you need
to include the result of `argmax` or even the probability distribution as we did in the example
above.

</exercise>


<exercise id="2" title="Looking at model outputs">

Although the `forward()` method is the most important (and often the only) method you need to
implement for your model, AllenNLP `Models` also implement other methods that make it easier to
interact with them.

## Post-processing with make\_output\_human\_readable()

The `make_output_human_readable()` method is one of them. `forward()` typically returns tensors and
is meant for machine consumption.  It runs in the inner loop of a training process, and taking the
GPU / CPU cycles to convert outputs to human readable format in this method is not a good idea.
`make_output_human_readable()` does what the name suggests—it takes the dictionary returned by the
`forward()` method and does whatever you need in order to make the output presentable to a person,
e.g., in a demo.  Most often this does vocabulary lookups on token or label ids; sometimes it also
does some kind of decoding or inference.  Our [prediction pipeline](/demos-and-predictors) calls
this method by default, as does `forward_on_instance`, which we discuss next.

## Making predictions with forward\_on\_instance(s)

There are two model methods—`forward_on_instance()` and `forward_on_instances()`—that come in handy
when you run inference using your model. Both take instance(s) instead of batched tensors, convert
them to indices and batch them on the fly, and run the forward pass (including the
`make_output_human_readable()` method). Before returning the results, these methods also convert any
`torch.Tensors` in the result dictionary to numpy arrays and separate the batched tensors into a
list of individual dictionaries per instance. As you might have guessed, `forward_on_instance()`
runs inference on a single instance, while `forward_on_instances()` does it with multiple instances
by batching them. These methods are used by `Predictors`.

<codeblock source="part2/building-your-model/model_prediction" setup="part2/building-your-model/setup_model_forward"></codeblock>

## Getting metrics with get_metrics()

To keep track of how a model is performing during training and validation, AllenNLP uses `Metrics`.
`Models` hold `Metric` objects that compute classification accuracy, BLEU score, SQuAD F1, or
whatever metric your model needs, and get updated at each call to the forward method.  AllenNLP's
`Trainer` calls your model's `get_metrics()` method to retrieve the computed metrics for, e.g.,
monitoring and early stopping. The method returns a dictionary of metric values associated with
their names. These values are usually returned by the `get_metric()` method of your `Metric`
objects.

The method receives a boolean flag `reset`, which indicates whether to reset the internal
accumulator inside the `Metric` objects. This flag is usually set at the end of each epoch, allowing
metrics to be computed over the entire epoch by accumulating statistics.

For a quick example of how to use this in your model, we'll repeat some of what we said in the
[quick start chapter](/training-and-prediction#3).  To integrate with the metrics code in our
training loop, you'll have to add code in three places.  First, your model constructor must
instantiate whatever metrics you want to compute:

<pre data-line="5" class="language-python line-numbers"><code>
class SimpleClassifier(Model):
    def __init__(self, ...):
        super().__init__(vocab)
        ...
        self.accuracy = CategoricalAccuracy()
</code></pre>

Second, in the forward pass, you need to call the metric, which accumulates statistics for the
current batch:

<pre data-line="8" class="language-python line-numbers"><code>
class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        self.accuracy(logits, label)
        ...
</code></pre>

And lastly, you'll need to implement the `get_metrics` method, which calls `get_metric` on each
`Metric` object that you are using:

```python
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
```

AllenNLP has a large number of metrics [built
in](https://docs.allennlp.org/main/api/training/metrics/metric/), and even more in our [model
library](https://github.com/allenai/allennlp-models).  If you don't see what you need there, you can
implement your own subclass of `Metric`.


</exercise>


<exercise id="3" title="Saving and loading a model">

## Saving your model

Oftentimes, you want to save and load trained models on disk. This is where using AllenNLP's
configuration files is very useful, because all we need to load a model, including weights,
configuration, and vocabulary, can be stored in a single tar file.  In this section we will discuss
saving and loading models that have been trained with configuration files; see our [chapter on
writing your own scripts](/writing-your-own-script#3) for pointers on how to save and load models if
you aren't using configuration files.

In order to properly serialize and restore an AllenNLP model, we need three things:

* Model config (specifications used to train the model)
* Model weights (trained parameters of the model)
* Vocabulary

In AllenNLP, model config is managed by the [`Params` class](/using-config-files) and can be saved
to disk using the `to_file()` method. You can retrieve the model weights using `model.state_dict()`
and save them to disk using PyTorch's `torch.save()`. The `Vocabulary.save_to_files()` method
serializes a `Vocabulary` object to a directory.

Because it is cumbersome to deal with these three elements every time you need to save, load, and
move around your model, AllenNLP provides utility functions for archiving and unarchiving your model
files. You can package up the model config, weights, and the vocabulary into a single `tar.gz` file,
along with any additional supplementary files, using the [`archive_model()`
method](http://docs.allennlp.org/main/api/models/archival/#archive_model).  This method assumes
that you trained a model using our training loop, and packages up the files that were saved while
the training loop was running.  Our training loop also calls this to package up the best model
weights when training is finished, so it is unlikely that you will need to call this method
yourself.

## Loading your model

In order to restore your model from files, you can use the
[`Model.load()`](http://docs.allennlp.org/main/api/models/model/#load) class method. It takes a
`Params` object which contains the model config, as well as the directory path where the model
weights and the vocabulary are serialized. The method also loads and restores the vocabulary.

Alternatively, you can simply use
[`load_archive()`](http://docs.allennlp.org/main/api/models/archival/#load_archive) to restore the
model from an archive file. This returns an
[`Archive`](http://docs.allennlp.org/main/api/models/archival/#archive) object, which contains the
config and the model.

In the example code below, we save and load our model using the two methods explained above, and
make sure the predictions stay the same before and after the serialization.

<codeblock source="part2/building-your-model/model_io" setup="part2/building-your-model/setup_model_io"></codeblock>

In practice, as long as you use AllenNLP commands (for example, `allennlp train`), model archiving
is automatically taken care of. When the training finishes or gets interrupted, the command
automatically saves the best model to a `model.tar.gz` file. You can also resume training from a
serialized directory.

## Model.from_archive

In addition to `load_archive()`, AllenNLP provides a convenience method `Model.from_archive()`.
This basically just calls `load_archive()` under the hood.  Its main purpose is that it is
registered as a `Model` constructor with type `"from_archive"`, so that you can load a saved model
from an archive file and continue training it using the `allennlp train` command.  To do this, you
would put the following snippet in your training configuration file:


```js
{
    ...
    "model": {
        "type": "from_archive",
        "archive_file": "/path/to/saved/archive/file.tar.gz"
    }
    ...
}
```

</exercise>


<exercise id="4" title="Initialization and regularization">

## Initialization

There are some cases where you want to initialize model parameters in a specific way (for example,
using [Xavier
initialization](https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/b71ac1e9fb49420d13e084ac67254a0bbd40f83f)).
AllenNLP provides a convenient abstraction for including initialization logic in the model
constructor that makes it easy to apply specific initializations based on regex matches on parameter
names.

In order to initialize individual model parameters, you use `Initializers` in AllenNLP, which are
basically just Python methods that take a tensor of parameters and apply some specific operations to
it. In most cases they are simply thin wrappers around PyTorch's initializers (the methods you can
find in `torch.nn.init`) that curry their parameters (like `mean` and `std`), making them
single-argument functions that are applied to tensors. AllenNLP provides a convenient abstraction
that allows you to apply them to your whole model.

You can instantiate specific `Initializers` manually inside a `Model` constructor if you wish,
though if you are doing that you might as well just call the `torch.nn.init` functions directly.  In
the spirit of separating code from configuration (as is a [common design principle in
AllenNLP](/using-config-files#1)), we provide a class called `InitializerApplicator` which, given a
list of regexes and their corresponding `Initializers`, applies initialization based on regex
matches on model parameter names. AllenNLP `Models` typically take such an initializer as a
constructor argument, so that the initialization can be easily configured.  Below is a constructor
call that instantiates an `InitializerApplicator`:

```python
applicator = InitializerApplicator(
    regexes = [
        ("parameter_regex_match1", NormalInitializer(mean=0.01, std=0.1)),
        ("parameter_regex_match2", UniformInitializer())],
    prevent_regexes = ["prevent_init_regex"])
```

The first parameter is a list of `(regex, initializer)` where `regex` is the regular expression that
gets matched against model parameter names, and `initializer` is an `Initializer` used for
initialization. You can also specify a list of regexes as `prevent_regexes`, in which case any
parameter matching the regex will be prevented from being initialized. In the code example below, we
create a toy model and initialize its parameter in two different ways—using individual
`Initializers` and using an `InitializerApplicator`.

<codeblock source="part2/building-your-model/model_init"></codeblock>

Note that it is your job to make sure that your model takes an `InitializerApplicator` as one of its
parameters and it is properly invoked during initialization. This can be achieved by adding the
following two lines to the model constructor:

<pre data-line="6,10" class="language-python line-numbers"><code>
class YourModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        ...
        initializer: InitializerApplicator = InitializerApplicator()
    ) -> None:
        super().__init__(vocab)
        ...
        initializer(self)
</code></pre>

## Regularizer

Regularization in AllenNLP works in a similar way to how initialization works.  AllenNLP provides an
abstraction called `Regularizers`, which are thin wrappers around Python methods that calculate and
return the penalty term (a scalar tensor) given model parameters.

In many cases, you may want to apply the penalty only to part of your model (for example, apply an
L2 penalty only to weights but not to biases). AllenNLP contains a `RegularizerApplicator` class,
which is analogous to `InitializerApplicator`, except it returns the penalty term instead of
modifying the model parameters. `RegularizerApplicator`, given a list of regexes and their
corresponding `Regularizers`, applies regularization based on regex matches on model parameter names
and returns the sum of all the computed penalties. Below is a constructor call that instantiates an
`RegularizerApplicator`:

```python
applicator = RegularizerApplicator(
    regexes=[
        ("parameter_regex_match1", L1Regularizer(alpha=.01)),
        ("parameter_regex_match2", L2Regularizer())
    ])
```

This applies an L1 penalty (with `alpha=0.01`) to all model parameters that match
`"parameter_regex_match1"`, and an L2 penalty (with its default parameters) to ones that match
`"parameter_regex_match2"`. `Model` returns the penalty term computed by a regularizer applicator
via the `get_regularization_penalty()` method, which is then used by the trainer and added to the
loss.

All you need to do to make your model support regularizers is to make sure the model constructor
takes a `**kwargs` argument and passes it to the superclass by calling `super().__init__(**kwargs)`
in the constructor. Then, when you construct your model (either in python code or through a
configuration file), pass a `RegularizerApplicater` parameter with the name `regularizer`.  In this
way, the `RegularizerApplicator` is automatically passed over to the base `Model`, where it is used
to compute regularization in `get_regularization_penalty()`.

In the following code, we create a toy model and obtain the regularization term in two different
ways—using individual `Regularizers` and using a `RegularizerApplicator`:

<codeblock source="part2/building-your-model/model_regularization"></codeblock>

Finally, note that you can also choose to take the regularization in your model by computing the
penalty term yourself and adding it to the loss in the forward method. The abstractions we covered
so far are a convenient way to get the regularization for all models automatically by default, by
having it available in the super class. This requires using AllenNLP's default trainer. If you are
using a different trainer than the built-in one, you need to put the regularization in your
`Model.forward()` method.

</exercise>
