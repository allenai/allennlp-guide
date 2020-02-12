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

<codeblock source="building-your-model/model_forward" setup="building-your-model/setup"></codeblock>

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

<codeblock source="building-your-model/model_predict" setup="building-your-model/setup"></codeblock>

## Getting metrics with get_metrics()

[As you learned before](/training-and-prediction#2), `Models` hold `Metrics` objects that keep track of related statistics which get updated by every call to the forward method. AllenNLP's `Trainer` calls your model's `get_metrics()` method to retrieve the computed metrics for, e.g., monitoring and early stopping. The method returns a dictionary of metric values associated with their names. These values are usually returned by the `get_metric()` method of your `Metric` objects.

The method receives a boolean flag `reset`, which indicates whether to reset the internal accumulator of `Metric` objects. This flag is usually set at the end of each epoch, allowing metrics to be computed over the entire epoch by accumulating statistics.

</exercise>

<exercise id="3" title="Saving and loading a model">

* Model.load()
* Model archiving
* exercise

</exercise>

<exercise id="4" title="Regularization and initialization">

</exercise>
