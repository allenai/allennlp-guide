---
title: 'Building your model'
description: 'This chapter gives a deep dive into one of the most important components of AllenNLP—Model—and an in-depth guide to building your own model.'
type: chapter
---

<textblock>

Models play a central role in AllenNLP and NLP applications in general. They implement the computation performed to transform input to output, and hold parameters that get optimized through training. In this chapter, we'll give a deep dive into inner workings of AllenNLP `Models`, and provide an in-depth guide to building, using, saving and loading your own model.

</textblock>

<exercise id="1" title="Model and Model.forward()">

AllenNLP uses the class `Model` to implement NLP models. `Model` subclasses  `torch.nn.Module`, meaning every AllenNLP `Model` is also a PyTorch `Module` and you can use it as any other PyTorch `Modules`, e.g., calling a model (which invokes the `__call__()` method) to run the forward pass, using `to()` to move it between devices, combining submodules to implement a larger model, and so on.

There are, however, a few key differences between PyTorch `Modules` and AllenNLP `Models`. The most important is the fact that `forward()` returns a dictionary, unlike PyTorch `Modules`, which usually return a tensor. AllenNLP `Models` also implement additional features for running and processing model predictions, saving and loading models, getting metrics, etc., which we'll elaborate on below.

The input/output spec of `Model.forward()` method is somewhat more strictly defined than that of PyTorch modules. Its parameters need to match field names in your data code exactly, [as we mentioned previously](/reading-textual-data#1). Instances created by the dataset reader are batched and converted to a set of tensors by AllenNLP. These batched tensors get passed to `Model.forward()` by their original field names. The following figure shows this process:

<img src="/part2/building-your-model/forward.svg" alt="Model.forward() and its parameters" />

Note that some fields (for example, training labels) may be absent during prediction. Make sure optional `forward()` parameters have default values as shown below, otherwise you'll get a missing argument error.

```python
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
```

In the exercise below, we create a toy model and observe how instances get batched and passed to the `forward()` method.

<codeblock source="part2/building-your-model/model_forward" setup="part2/building-your-model/setup_model_forward"></codeblock>

As mentioned above, `Model.forward()` returns a dictionary, instead of a tensor. Although technically you can put anything you want in this dictionary, if you want to train your model through backpropagation using our `Trainer`, the return value must contain a `"loss"` key pointing to a scalar `Tensor` that represents the loss, which then gets minimized by the optimizer. For example, here's the snippet from the forward method of `SimpleClassifier`, where a dictionary consisting of `"probs"` and an optional `"loss"` is returned:

```python
probs = torch.nn.functional.softmax(logits)
output = {'probs': probs}
if label is not None:
    self.accuracy(logits, label)
    output['loss'] = torch.nn.functional.cross_entropy(logits, label)
return output
```

You also need to include in this dictionary any information necessary for subsequent analysis, visualization, and so on. For example, if you want to show the predicted labels in a demo, you need to include the result of `argmax` or even the probability distribution as we did in the example above.

</exercise>

<exercise id="2" title="Looking at model outputs">

Although the `forward()` method is the most important (and often the only) method you need to implement for your model, AllenNLP `Models` also implement other methods that make it easier to interact with them.

## Post-processing with decode()

The `decode()` method is one of them. It takes the dictionary returned by the `forward()` method and runs whatever post-processing you need for your model. This is often some sort of decoding or inference, but not necessarily. In a common scenario, `forward()` returns logits or probabilities, then `decode()` takes its result and runs some sort of beam search or constrained inference. The method can also reference `Vocabulary` in order to convert indices back to string representations (as we'll see shortly).

## Making predictions with forward\_on\_instance(s)

There are two model methods—`forward_on_instance()` and `forward_on_instances()`—that come in handy when you run inference using your model. Both take instance(s) instead of batched tensors, convert them to indices and batch them on the fly, and run the forward pass (including the `decode()` method). Before returning the results, these methods also convert any `torch.Tensors` in the result dictionary to numpy arrays and separate the batched tensors into a list of individual dictionaries per instance. As you might have guessed, `forward_on_instance()` runs inference on a single instance, while `forward_on_instances()` does it with multiple instances by batching them. These methods are used by `Predictors`.

<codeblock source="part2/building-your-model/model_predict" setup="part2/building-your-model/setup"></codeblock>

## Getting metrics with get_metrics()

[As you learned before](/training-and-prediction#2), `Models` hold `Metrics` objects that keep track of related statistics which get updated by every call to the forward method. AllenNLP's `Trainer` calls your model's `get_metrics()` method to retrieve the computed metrics for, e.g., monitoring and early stopping. The method returns a dictionary of metric values associated with their names. These values are usually returned by the `get_metric()` method of your `Metric` objects.

The method receives a boolean flag `reset`, which indicates whether to reset the internal accumulator of `Metric` objects. This flag is usually set at the end of each epoch, allowing metrics to be computed over the entire epoch by accumulating statistics.

</exercise>

<exercise id="3" title="Saving and loading a model">

## Saving your model

Oftentimes, you want to save and load trained models on disk. In order to properly serialize and restore an AllenNLP model, you need to take care of the following three things:

* Model config (specifications used to train the model)
* Model weights (trained parameters of the model)
* Vocabulary

In AllenNLP, model config is managed by the `Params` class (which we'll cover more in depth [later](/using-config-files)) and can be saved to disk using the `to_file()` method. You can retrieve the model weights using `model.state_dict()` and save them to disk using PyTorch's `torch.save()`. The `Vocabulary.save_to_files()` method serializes a `Vocabulary` object to a directory.

Because it is cumbersome to deal with these three elements every time you need to save, load, and move around your model, AllenNLP provides utility functions for archiving and unarchiving your model files. You can package up the model config, weights, and the vocabulary into a single `tar.gz` file, along with any additional supplementary files, using the `archive_model()` method.

## Loading your model

In order to restore your model from files, you can use the `Model.load()` class method. It takes a `Params` object which contains the model config, as well as the directory path where the model weights and the vocabulary are serialized. The method also loads and restores the vocabulary.

Alternatively, you can simply use `load_archive()` to restore the model from an archive file. This returns an `Archive` object, which contains the config and the model.

In the example code below, we save and load our model using the two methods explained above, and make sure the predictions stay the same before and after the serialization.

<codeblock source="part2/building-your-model/model_io" setup="part2/building-your-model/setup_model_io"></codeblock>

In practice, as long as you use AllenNLP commands (for example, `allennlp train`), model archiving is automatically taken care of. When the training finishes or gets interrupted, the command automatically saves the best model to a `model.tar.gz` file. You can also resume training from a serialized directory.

</exercise>

<exercise id="4" title="Initialization and regularization">

## Initialization

There are some cases where you want to initialize model parameters in a specific way (for example, using [Xavier initialization](https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/b71ac1e9fb49420d13e084ac67254a0bbd40f83f)). In vanilla PyTorch, you need to include your initialization logic in your model constructor, or you need to make sure it's properly initialized after creating the model. AllenNLP provides a convenient abstraction that allows you to inject an initialization logic using model config. 

AllenNLP uses `Initializers` to initialize model parameters, which are basically just Python methods that take a tensor of parameters and apply some specific operations to it. 

`Initializers` can be instantiated via the `Initializer.from_params()` class method and can be used to initialize specific model parameters, although in reality you rarely need to deal with individual `Initializers` yourself. Instead, AllenNLP provides a class called `InitializerApplicator` which, given a list of regexes and their corresponding `Initializers`, applies initialization based on regex matches on model parameter names. Below is a config that instantiates an `InitializerApplicator`:

```
"initializer": [
    ["parameter_regex_match1",
        {
            "type": "normal",
            "mean": 0.01,
            "std": 0.1
        }
    ],
    ["parameter_regex_match2", "uniform"]
    ["prevent_init_regex", "prevent"]
]
```

It takes a list of `(regex, initializer)` where `regex` is the regular expression that gets matched against model parameter names, and `initializer` is the config used to instantiate an `Initializer`. You can also specify a special keyword `"prevent"` or `{"type": "prevent"}` as the initializer config, in which case any parameter matching the regex will be prevented from being initialized. In the code example below, we create a toy model and initialize its parameter in two different ways—using individual `Initializers` and using an `InitializerApplicator`.

<codeblock source="part2/building-your-model/model_init"></codeblock>

Note that it is your job to make sure that your model takes an `InitializerApplicator` as one of its parameters and it is properly invoked during initialization. This can be achieved by adding the following two lines to the model constructor:

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

Regularization in AllenNLP works in a similar way to how initialization works. AllenNLP provides an abstraction called `Regularizers`, which are thin wrappers around Python methods that calculate and return the penalty term (a scalar tensor) given model parameters. 

In many cases, you may want to apply the penalty only to part of your model (for example, apply an L2 penalty only to weights but not to biases). AllenNLP implements the `RegularizerApplicator` class, which works in a very similar way to `InitializerApplicator`, except the former returns the penalty term instead of modifying the model parameters. `RegularizerApplicator`, given a list of regexes and their corresponding `Regularizers`, applies regularization based on regex matches on model parameter names and returns the sum of all the computed penalties. Below is a sample config that instantiates an `RegularizerApplicator`:

```
"regularizer": [
    ["parameter_regex_match1", {"type": "l1", "alpha": 0.01}],
    ["parameter_regex_match2", "l2"]
]
```

This applies an L1 penalty (with `alpha=0.01`) to all model parameters that match `"parameter_regex_match1"`, and an L2 penalty (with its default parameters) to ones that match `"parameter_regex_match2"`. `Model` returns the penalty term computed by a regularizer applicator via the `get_regularization_penalty()` method, which is then used by the trainer and added to the loss.

You need to make sure that your model takes an `RegularizerApplicator` as a constructor parameter and it is passed to the parent class, where `get_regularization_penalty()` is implemented. You can do this by modifying your model definition as follows:

<pre data-line="6,8" class="language-python line-numbers"><code>
class YourModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        ...
        regularizer: Optional[RegularizerApplicator] = None
    ) -> None:
        super().__init__(vocab, regularizer)
        ...
</code></pre>

In the following code, we create a toy model and obtain the regularization term in two different ways—using individual `Regularizers` and using a `RegularizerApplicator`:

<codeblock source="part2/building-your-model/model_regularization"></codeblock>

</exercise>
