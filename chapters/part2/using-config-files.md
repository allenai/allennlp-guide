---
title: 'Using config files: FromParams and Registrable'
description: "This chapter describes AllenNLP's simple dependency injection framework."
type: chapter
---

<exercise id="1" title="Motivation: Dependency Injection">

A core motivating principle of AllenNLP (and object-oriented software design generally) is to
separate the configuration of an object from its implementation code.  This is accomplished with
[_dependency injection_](https://www.jamesshore.com/Blog/Dependency-Injection-Demystified.html),
which is a fancy term that in practice basically just means "objects take all of their instance
variables as constructor parameters".

We write our code using high-level abstractions, which lets us make changes to lower-level details
by just changing the constructor parameters that we pass to `Models`, `DatasetReaders`, etc.  It's
the job of the final script that runs things to actually configure the behavior of the code by
creating all of the objects that get passed to these constructors.

It just fine to have your final script be python code that manually constructs a bunch of objects,
passes them to where they need to be, and calls a training loop.  Everything in AllenNLP still works
if you want to completely forget about configuration files (and we have a chapter on [setting up
your own training loop](/TODO) that goes into detail on how to do this).  But, we find it very nice
to be able to put all of the configuration into a simple JSON file that specifies an entire
experiment that was run.  If you have these configuration files, you can organize them, save them,
check them into a git repository, or whatever else you want, and easily know exactly what
configuration corresponded to a particular experiment.  It also makes it much easier to pass around
a single tar file that contains everything you need to run a trained model.

This is a pattern for keeping track of controlled scientific experiments that we really like, so we
built a light-weight dependency injection framework to make this easy.  This chapter explains what
it is and how to use it.

</exercise>

<exercise id="2" title="Injecting dependencies with FromParams">

If you've heard of dependency injection before or used a framework like
[Guice](https://github.com/google/guice), you might be cringing at having to learn something new and
complicated.  But we think those are too complicated, too.  Our framework, based on the `FromParams`
class, is very simple.  All we do is match type annotations on constructor arguments to parameter
dictionaries loaded from a JSON file.  That's it.  Its utility comes from handling parameterized
collections, user-defined types, and polymorphism, and operating recursively.

Under the hood, all that FromParams is doing is constructing arguments for an object from JSON, but,
one you get the hang of it, this simple functionality enables you to write and modify reproducible
experiments with ease.

To construct a class from a JSON dictionary, it has to inherit from the `FromParams` abstract base
class. For example, say that we want to create an object to represent a Gaussian distribution and
include it in our NLP experiments. We would first write a class like the following:

```python
from allennlp.common import FromParams

class Gaussian(FromParams):
    def __init__(self, mean: float, variance: float):
        self.mean = mean
        self.variance = variance
```

Importantly, we have to use type annotations for the constructor arguments: this tells FromParams
how to parse the JSON in the configuration file (and we strongly recommend using type annotations
anyway, as it greatly improves the readability of your code). Once we have done this, we can create
the object from a JSON dictionary instead of using its constructor:

```python
from allennlp.common import Params
import json

param_str = """{"mean": 0.0, "variance": 1.0}"""
params = Params(json.loads(param_str))
gaussian = Gaussian.from_params(params)
```

The `.from_params()` method (provided by `FromParams`) looks in the parameter dictionary and matches
keys in that dictionary to argument names in `Gaussian.__init__`.  When it finds a match, it looks
at the type annotation of the argument in `__init__` and constructs an object of the appropriate
type from the corresponding value in the parameter dictionary.  In this case, both arguments are
`floats`, so that construction is easy.

If you're only using this to configure a Gaussian, this might seem like severe overkill, and you're
right; the above code does the same thing as the much simpler:

```python
gaussian = Gaussian(0.0, 1.0)
```

But `FromParams` can do much more than pull floats out of dictionaries; it can also handle
user-defined types, which are constructed recursively.  For example, we can use our `Gaussian`
class as a type annotation on another class that inherits from `FromParams`:

```python
class ModelWithGaussian(FromParams):
    def __init__(self, gaussian: Gaussian):
        self.gaussian = gaussian
        ...
```

We can now construct a `ModelWithGaussian` using `.from_params()` as follows:

```python
param_str = """{"gaussian": {"mean": 0.0, "variance": 1.0}}"""
params = Params(json.loads(param_str))
model = ModelWithGaussian.from_params(params)
```

The nested dictionary gets passed to the `Gaussian`, which is recursively constructed just as we saw
above.

In practice, what we do in AllenNLP is have _all_ of our classes inherit from `FromParams`, from our
`DatasetReaders`, `Models` and `Trainer` to our training loop itself, so that we can have one file
that specifies all of the configuration that goes into an entire experiment.  Everything is built
recursively from the specification in a file that looks something like this:

```json
{
  "dataset_reader": {
    // ...
  },
  "model": {
    "type": "model_with_gaussian",
    "gaussian": {
      "mean": 0.5,
      "variance": 0.3
    }
  },
  "trainer": {
    // ...
  },
  // ...
}
```

We'll get into more detail on how all of the bells and whistles in `FromParams` work in the rest of
sections in this chapter.  For now, we'll end this section with a more complex, real example of
utilizing `FromParams` from the AllenNLP internals: the
[`FeedForward`](https://github.com/allenai/allennlp/blob/master/allennlp/modules/feedforward.py)
class. `FeedForward` is a `torch.nn.Module` representing a feedforward neural network. `FeedForward`
extends `FromParams` so that options like the number of layers, layer widths, and activation
functions can be configured directly from the config file:

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
        ...
```

There are a few things to notice here. First, the type `Activation` itself extends `FromParams`:
when we use it as a type annotation, `FromParams` will parse parameters recursively to guarantee
that `FeedForward` receives a correctly configured `Activation` object! Second, `FromParams` is able
to correctly construct arguments with most of the type annotations in python's built-in `typing`
library.  This includes `List` (which is specified in a configuration file as a JSON list), `Union`
(`FromParams` will try each type in turn with the given parameter dictionary and keep the first one
that succeeds), and `Dict` (which is specified in a configuration file as a JSON dictionary), among
others.  Lastly, `FromParams` handles default values for constructor argumnts just fine; when there
is no correspond key in a parameter dictionary for a particular argument, the default value is used.
If there is no default value, `FromParams` will crash, complaining that a key was missing in the
configuration file.  Similarly, if there are _extra_ parameters in a configuration file, where a
provided key does not match the name of a constructor argument, `FromParams` will also crash.

</exercise>

<exercise id="3" title="Handling polymorphism with Registrable">

Another fundamental design principle in AllenNLP is the use of
[polymorphism](https://www.programiz.com/python-programming/polymorphism) to abstract away low-level
details of data processing or model operations.  We encapsulate common operations in abstract base
classes, then write code using those abstract base classes instead of concrete instantiations.

For example, we might want to write a part of speech tagging model such that it only states that
sequences are encoded in some way, not the specific way they are encoded.  Then the same code can
easily be configured to use either an LSTM or a convolutional layer to encode the sequences,
depending on what concrete instantiation the model is passed.  Our chapter on [common modeling
architectures](/common-architectures) has many more examples of these abstractions and how they are
used.

Using these abstract base classes as type annotations presents a challenge to constructing objects
using `FromParams`—we need a way of knowing which concrete instantiation to construct, from the
configuration file.  We handle this sort of polymorphic dependency injection using a class called
`Registrable`.

When we create an abstract base class, like `Model`, we have it inherit from `Registrable`.
Concrete subclasses then use a decorator provided by `Registrable` to make themselves known to the
base class: `@Model.register("my_model_name")`.  Then, when we are trying to construct an argument
with a type annotated with the base `Model` class, `FromParams` looks for a `type` key whose value
must correspond to some registered name.

Let's look at a concrete example of this.  `Seq2VecEncoder` is a `Registrable` type, representing
the operation of mapping a sequence of hidden states to a single vector. Specific concrete
implementations extend `Seq2VecEncoder` and register themselves as `Seq2VecEncoders` using the
decorator `@Seq2VecEncoder.register("encoder_name")`. For example, the following code snippet comes
from the `CnnEncoder`, which runs a convolutional neural network to encode the sequence into a
vector:

```python
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

@Seq2VecEncoder.register("cnn")
class CnnEncoder(Seq2VecEncoder):

    def __init__(self,
        embedding_dim: int,
        num_filters: int,
        ngram_filter_sizes: Tuple[int, ...]
        conv_layer_activations: Activation = None,
        output_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        ...
```

The type name `cnn` can then be used to construct instances of `CnnEncoder` from JSON config files:

```json
{
  // ...
  "encoder": {  // An argument with a Seq2VecEncoder type annotation
    "type": "cnn",  // Tells FromParams to use CnnEncoder as the concrete Seq2VecEncoder
    "embedding_dim": 64,  // All of the rest of the parameters match constructor arguments
    "num_filters": 128,
    ...
  },
  // ...
}
```

You can straightforwardly adapt the pattern above to create your own concrete implementations of
various built-in `Registrables` (such as `Seq2VecEncoders`).  When you do this, you need to make
sure that `FromParams` knows about them, which we discuss in a [later section of this
chapter](/using-config-files#8).

If you're feeling adventurous, you can also create your own `Registrable` abstract type.  You might
want to do this if you have an operation that you want to encapsulate and experiment with, but we
don't already have an abstraction for it.  To make your own, simply extend `Registrable` and use
your new class as you would any other registry. For example:

```python
from allennlp.common import Registrable

class Muppet(Registrable):
    pass

@Muppet.register("elmo")
class Elmo(Muppet):
    pass
```

Then you could create classes that use `Muppet` as type annotations, and instantiate them from
configuration files.

</exercise>

<exercise id="4" title="Mixing configured and non-configured objects">

What about objects that aren't constructed using `FromParams`?

Talk about how `extras` works (no need to use that term).  E.g., this is how `vocab` gets passed
to the model, where the trainer does `Model.from_params(params=params.pop("model"), vocab=vocab)`.

</exercise>

<exercise id="5" title="Avoiding duplication of base class parameters">

What about abstract base class parameters?

Talk about `**kwargs`, and how we inspect super class constructors to figure out what's available,
so we can construct those too.  This allows adding parameters to base classes without changing
subclass code.

</exercise>

<exercise id="6" title="Allowing multiple constructors for the same object">

What about having multiple ways of constructing an object?

Talk about `Vocabulary`, how you can register things multiple times with different ways of
constructing the object.

</exercise>

<exercise id="7" title="Handling sequential dependencies between constructor arguments with Lazy">

What about sequential dependencies between arguments?

`Lazy` and how it's used.  Mention somewhere in here that `Lazy.construct()` can be called
multiple times if necessary.

</exercise>

<exercise id="8" title="Including your own registered components">

Using these components in your code

`--include-package` and the plugins stuff.  You can write your own classes that inherit from
`Registrable` or `FromParams`, adding components or entire new abstractions (you already mentioned
examples for this).

You can easily write your own classes implementing FromParams and utilize them in AllenNLP configs.
There is one important caveat here though: you will need to run your allennlp command with the
`--include-package=<YOUR_PACKAGE>` flag so that the library can find and register your custom
classes.

Also plugins.

</exercise>
