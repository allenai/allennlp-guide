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

The input/output spec if `Model.forward()` method is somewhat more strictly defined than that of PyTorch modules. Its parameters need to match field names exactly, [as we mentioned previously](/reading-textual-data#1). Instances created by the dataset reader are batched and converted to a set of tensors by AllenNLP. These batched tensors get passed to `Model.forward()` by their original field names. The following figure shows this process:

<img src="/part2/building-your-model/forward.svg" alt="Model.forward() and its parameters" />

Note that some fields (for example, training labels) may be absent during prediction. Make sure optional `forward()` parameters have default values as shown below, otherwise you'll get a missing argument error.

```python
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
```

In the exercise below, we create a toy model and observe how instances get batched and passed to the `forward()` method.

<codeblock source="building-your-model/model_forward"></codeblock>

As mentioned above, `Model.forward()` returns a dictionary, instead of a tensor. Although technically you can put anything you want in this dictionary, if you want to train your model through backpropagation, the return value must contain a `"loss"` key pointing to a scalar `Tensor` that represents the loss, which then gets minimized by the optimizer. For example, here's the snippet from the forward method of `SimpleClassifier`, where a dictionary consisting of `"probs"` and a optional `"loss"` is returned:

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

* Other Model methods
    * forward_on_instance(s)
    * decode (here's another place vocab is used)
    * exercise
    * get_metrics

</exercise>

<exercise id="3" title="Saving and loading a model">

* Model.load()
* Model archiving
* exercise

</exercise>

<exercise id="4" title="Regularization and initialization">

</exercise>
