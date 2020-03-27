---
title: 'Using config files: `Registrable` and `FromParams`'
description: 'This chapter explains how `FromParams` and `Registrables` work in AllenNLP.'
type: chapter
---

<exercise id="1" title="part2/using-config-files/from-params`">

## Configuring Experiments with `FromParams`

FromParams is a simple framework for <i>dependency injection</i> that allows you to separate your code from the configuration of experiments. It allows you to use a single JSON file to define an experiment, configuring model and task settings. This encourages good research hygiene, so you have one config file per run of a model, and can easily see what happened. Under the hood, all that FromParams is doing is constructing arguments for an object from JSON, but, one you get the hang of it, this simple functionality enables you to write and modify reproducible experiments with ease.

To make a type configurable, we make it extend the `FromParams` abstract base class. For example, say that we want to create an object to represent a Gaussian distribution and include it in our NLP experiments. We would first write a class like the following:

```python
from allennlp.common.from_params import FromParams

class Gaussian(FromParams):
	def __init__(self, mean: float, variance: float):
		self.mean = mean
		self.variance = variance
```

Importantly, we have to use type annotations for the constructor arguments: this tells FromParams how to parse the JSON in the configuration file. Once we have done this, other objects can utilize a configurable `Gaussian` simply by annotating one of their constructor arguments with the `Gaussian` type. For example:

```python
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

@Model.register("with_gaussian")
class ModelWithGaussian(Model):
	def __init__(self, vocab: Vocabulary, gaussian: Gaussian):
		super().__init__(vocab)
		self.gaussian = gaussian
```

The following code snippet would now construct a `ModelWithGaussian`:

```python
from allennlp.models import Model

model = Model.from_params(params={{"gaussian": {"mean": 0., "variance": 1.}}}, vocab=vocab)
```

More practically, though, a model containing a configurable `Gaussian` can be specified in a JSON configuration file as follows:

```json
{
  // ...
  "model": {
    "type": "with_gaussian",
    "gaussian": {
      "mean": 0.5,
      "variance": 0.3
    }
  },
  // ...
}
```

Note that `vocab` is an “extra” argument to the model class: it is not specified in the configuration file, but manually added to the parameter dictionary by the core AllenNLP logic for parsing configuration files.

Let’s look at an actual example utilizing `FromParams` from the AllenNLP internals: the [`FeedForward`](https://github.com/allenai/allennlp/blob/master/allennlp/modules/feedforward.py) class. `FeedForward` is a torch module representing a feedforward neural network. `FeedForward` extends `FromParams` so that options like the number of layers, layer widths, and activation functions can be configured directly from the config file:

```python
from typing import List, Union
from allennlp.nn.activations import Activation

class FeedForward(torch.nn.Module, FromParams):

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dims: Union[int, List[int]],
        activations: Union[Activation, List[Activation]],
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:
        super().__init__()
        # ...
```

There are a couple things to notice here. First, the type `Activation` is itself a `Registrable`: when we use it as a type annotation, AllenNLP will apply `FromParams` recursively to guarantee that the corresponding parameters are parsed correctly. Second, we can use the type annotations from the built-in typing library to support various kinds of structures in the configuration file: `List` corresponds to a JSON list, `Union` corresponds to an argument with multiple options for its type, and `Dict` corresponds to an argument that should be interpreted as a JSON dictionary.

You can write your own classes implementing FromParams and utilize them in AllenNLP configs! There is one important caveat here though: you will need to run your allennlp command with the `--include-package=<YOUR_PACKAGE>` flag so that the library can find and register your custom classes.

</exercise>

<exercise id="2" title="part2/using-config-files/registrables`">

### `Registrables`: Polymorphic Dependency Injection

Often when we are defining a machine learning model, it makes sense to utilize abstract components. For example, we might want to write our part of speech tagging model such that it can easily be configured to use either an LSTM or a convolutional layer to encode sequences. AllenNLP supports this sort of polymorphic dependency injection using the `Registrable` interface.

`Seq2SeqEncoder` is one example of a `Registrable` type, representing any abstract module that maps a sequence of hidden states to another hidden state sequence. Specific concrete implementations extend `Seq2SeqEncoder` and are assigned a type name by the decorator `Seq2SeqEncoder.register`. For example, the following code snippet comes from the `PassThroughEncoder`, which simply returns the input sequence:

```python
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder

@Seq2SeqEncoder.register("pass_through")
class PassThroughEncoder(Seq2SeqEncoder):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # …
```

The type name `pass_through` can then be used to construct specific instances of `PassThroughSeq2SeqEncoder` from JSON config files:

```json
{
  // ...
  "encoder": {  // A Seq2SeqEncoder component, for example within a simple_tagger.
    "type": "pass_through",
    "input_dim": 64
  },
  // ...
}
```

You can straightforwardly adapt the pattern above to create your own concrete implementations of various built-in Registrables (such as Seq2SeqEncoders). As with FromParams, you will need to run your allennlp command with  `--include-package=<YOUR_PACKAGE>` so the library can find and register your custom classes.

Another less common use case is creating a new Registrable abstract type. To do this, simply extend Registrable and use your new class as you would any other registry. For example:

```python
from allennlp.common import Registrable

class Muppet(Registrable):
	pass

@Muppet.register(“elmo”)
class Elmo(Muppet):
	pass
```

</exercise>
