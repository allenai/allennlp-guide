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
once you get the hang of it, this simple functionality enables you to write and modify reproducible
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
others.  Lastly, `FromParams` handles default values for constructor arguments just fine; when there
is no corresponding key in a parameter dictionary for a particular argument, the default value is
used.  If there is no default value, `FromParams` will crash, complaining that a key was missing in
the configuration file.  Similarly, if there are _extra_ parameters in a configuration file, where a
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


<exercise id="4" title="Handling runtime and sequential dependencies">

Sometimes arguments can't be constructed just from a configuration file, because they depend on
objects or data that was obtained at runtime.  There are two main cases where this happens in NLP
code:

* The first case is when data is read from a file, and `Instances` are created from that data.  In
  order to create a `Vocabulary`, we need to see those `Instances`, but this isn't something that
  can or should be specified in a configuration file somewhere.
* The second case is when an object has two dependencies that themselves depend on each other.  For
  example, our training loop is run by a `TrainModel` object, which takes as constructor parameters
  both a `Model` and a `Trainer`.  The `Trainer` itself takes the `Model` as a constructor
  parameter, so the `Model` has to be constructed _before_ the `Trainer` and then passed to the
  `Trainer`.

Handling these cases is the part of AllenNLP's `FromParams` framework where the most magic happens,
that is hardest to understand.  We'll try to make it clear in this section.

We'll start with how a `Model` gets its `Vocabulary`.  If you've seen any configuration files for
AllenNLP models, you'll know that the `"models": {}` section doesn't have a `"vocab"` key, even
though the `Model`'s constructor takes a `vocab` argument.  For example, the `SimpleClassifier` that
we built in [part 1 of this course](/your-first-model) had a constructor that looked like this:

```python
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        ...
```

but the configuration file that we used to train it had a `"model"` section that looked like this:

```js
{
    ...
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    ...
}
```

In other words, no `"vocab"` key corresponding to the `vocab` constructor argument.  We said
[earlier](#2) that this mismatch would cause `FromParams` to crash.  The reason that it doesn't
crash is that we pass in the `vocab` object as a separate parameter when we call
`Model.from_params` in our training loop.  `.from_params()` accepts arbitrary keyword arguments,
which are available to the objects that are being constructed.  So the actual call looks something
like `Model.from_params(params=params, vocab=vocab)`.  The example below lets you see how this works
in action.

<codeblock source="part2/using-config-files/extras_basic"></codeblock>

A tricky thing is that these objects are available _recursively_, to all objects that are created
while creating the original object.  In the `SimpleClassifier` that we mentioned above, this lets us
take the `Vocabulary` as an argument when constructing the `Embedding` layer of a model, which needs
to know how many embeddings to create.  Here's a toy example showing how this works:

<codeblock source="part2/using-config-files/extras_recursive"></codeblock>

The hard part about this functionality is that you have to know which parameters are passed this way
in a given training loop, and which ones must be specified in a configuration file.  There is no way
to programmatically know that, but there is a simple rule of thumb: if the constructor argument was
already specified higher up in the config file (or is derived from something that was specified
higher up in the file), then you don't need to repeat it.  So if you start from the top-level object
that's being constructed (which for `allennlp train` is a [`TrainModel`
object](https://docs.allennlp.org/master/api/commands/train/#trainmodel-objects)) and work your way
down, it should be relatively clear which keys are unnecessary.

To make it easier to figure this out, we have additionally tried to make sure our documentation
clearly states which parameters should not be present in configuration files for our training loop
(see, e.g., the `vocab` parameter in our [`Model`
documentation](https://docs.allennlp.org/master/api/models/model/#model-objects)).  If there's a
place where we failed to document this correctly, PRs to fix it are very welcome!  The major one
that you will typically worry about is the `vocab` that's passed to the model, but there are also
several parameters passed to the
[`Trainer`](https://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects)
that don't get entries in a configuration file.

### How to construct these objects

In the above discussion, we have talked about how these sequential dependencies are handled from the
perspective of the _object being constructed_—the `Model` that needs to get a `Vocabulary`.  Now we
move to discussing how these dependencies are handled from the perspective of the _object doing the
construction_—the `TrainModel` object that needs to pass the `Vocabulary` to the `Model`.

To handle cases where an object has two constructor parameters, where one of them needs to be passed
to the constructor of the other, AllenNLP introduces a `Lazy` class that gets special treatment in
`FromParams` when it is used as a type annotation.  `Lazy` has a `construct()` method that will
accept arbitrary keyword arguments and pass them on to the constructor of the object it's wrapping.
This means that we can specify _some_ arguments in a configuration file, store them in the `Lazy`
object, then finish constructing the object with a call to `construct()` later, once the rest of its
dependencies are created.

This is easiest to see with a simple example, so we'll show some example code here:

<codeblock source="part2/using-config-files/lazy_bad"></codeblock>

What we saw there was the `ModelWithGaussian` constructing a `Vocabulary` object itself, then
passing it to the lazily-constructed `Gaussian` object, which took some of its constructor
parameters from a `Params` object, and some from `Lazy.construct()`.

If you actually use `Lazy` as a type annotation in an `__init__` method, you're requiring everyone
who constructs your objects to wrap those arguments in `Lazy` objects first.  This makes your
objects pretty annoying to use without configuration files, so we don't recommend actually doing
this.  Instead, we recommend registering a [separate constructor](#6) that has the `Lazy`
annotations, keeping the standard `__init__` method free of these lazy objects.  Here's another
example that's set up in the way that we would actually recommend using `Lazy`:

<codeblock source="part2/using-config-files/lazy_good"></codeblock>

The use of `Lazy` can get arbitrarily nested, to create complex pipelines that are all determined by
configuration files.  For example, here's part of the
[`TrainModel`](https://docs.allennlp.org/master/api/commands/train/#trainmodel-objects) constructor:

```python
  def from_partial_objects(
      cls,
      dataset_reader: DatasetReader,
      train_data_path: str,
      model: Lazy[Model],
      data_loader: Lazy[DataLoader],
      trainer: Lazy[Trainer],
      vocabulary: Lazy[Vocabulary] = None,
      ...
  ):
```

What the logic inside this constructor does is first create the `DatasetReader`, which needs no
other dependencies.  It then loads the data in `train_data_path` to create a list of `Instances`,
which get passed to the `Vocabulary` constructor with `vocab =
vocabulary.construct(instances=instances)`, and to the `DataLoader` in the same way.  The vocabulary
then gets passed to the `Model` constructor with `model.construct(vocab=vocab)`, and then the model
and data loader get similarly passed to the `Trainer`, before calling `trainer.train()`.  You can
see this logic yourself by looking at [the
code](https://github.com/allenai/allennlp/blob/master/allennlp/commands/train.py) for this method.
If you want to implement a similar pipeline yourself, then that class would be a good example.
Otherwise, you don't need to worry about this, as this is advanced functionality that you typically
never need to use when doing model development.

</exercise>


<exercise id="5" title="Avoiding duplication of base class parameters">

When using abstract base classes such as `DatasetReader` and `Model`, it is often convenient to
provide functionality in the base class, where the functionality requires constructor arguments.
For example, our `DatasetReader` object has options for laziness, caching, and truncation:

```python
class DatasetReader(Registrable):
    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
    ) -> None:
```

If you want to get this functionality in your `DatasetReader` subclass, you need to pass these
arguments to the base class, which means you also need to take them as constructor parameters:

```python
class MyFancyDatasetReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        cache_directory: Optional[str] = None,
        max_instances: Optional[int] = None,
        my_other_args: SomeType = None,
        ...
    ) -> None:
        super().__init__(lazy, cache_directory, max_instances)
        ...
```

If you do this, then you can specify those parameters in a configuration file, just like any other
constructor parameter.  But, this means that if new functionality ever gets added to the base class,
your code has to change in order to take advantage of it.  Instead, you could write your constructor
using `**kwargs` as a parameter, so that you don't have to change your dataset reader code if new
parameters get added (you would still have to change your object creation code, though):

```python
class MyFancyDatasetReader(DatasetReader):
    def __init__(
        self,
        my_other_args: SomeType = None,
        ...
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        ...
```

AllenNLP's `FromParams` logic handles this case, allowing you to construct superclass parameters
when your constructor takes a `**kwargs` argument.  This is done by inspecting the superclass
constructor when `**kwargs` is encountered, and adding superclass parameters as additional arguments
to be constructed from the configuration file.  You can see an example of this in action in the
following code:

<codeblock source="part2/using-config-files/kwargs"></codeblock>

</exercise>


<exercise id="6" title="Allowing multiple constructors for the same object">

Occasionally, you want to have multiple ways of constructing the same kind of object, with disjoint
parameters in each case.  For example, our `Vocabulary` object has many ways in which it can be
constructed: from a collection of `Instances`, from saved files, from both, or from nothing at all.
We don't want different `Vocabulary` objects or subclasses for each of these, as the underlying
object is the same, the only difference is in how the object is constructed.

To accomplish this, `Registrable` allows specifying a constructor other than `__init__` to use when
constructing the object.  This currently only works with registered types, however, not with classes
that inherit directly from `FromParams`.  When registering your type, you include a
`constructor='some_method_name'` argumnt to the call to `register()`, where `'some_method_name'`
must be a `@classmethod` that returns an object of the registered type.  Here's a toy example to
demonstrate how this works:

<codeblock source="part2/using-config-files/multiple_constructors"></codeblock>

The same functionality also works if you want to register a single constructor to use, but have it
be separate from `__init__`.  You might want to do this if you have sequential dependencies that
require the use of `Lazy`, as we talked about in a [previous section](#4).

</exercise>


<exercise id="7" title="Including your own registered components">

We have several `Registrable` components in AllenNLP which you can write subclasses of.  You can
also write your own `Registrable` classes.  But in order for `Registrable` to be able to map the
strings that you use in your call to `@Component.register("registered_name")` to concrete classes to
construct, that code has to be run before the `FromParams` pipeline tries to construct the object.
The `register()` call gets run when the module containing it gets imported.  This means that if
you're using `allennlp train` to train your model, you need to have your modules get imported before
our main training loop tries to construct all of its required arguments.  There are two ways to do
this: `--include-package` and `.allennlp_plugins`.

Most `allennlp` commands accept an `--include-package` argument.  This is a module name that
`allennlp` will try to import at the beginning of its execution, so that whatever custom classes you
have written can register themselves.  You just need to be sure that the module that you're
importing recursively imports whatever other modules contain the calls to `.register()`.

Because remembering to include the `--include-package` flag for every command can be cumbersome,
AllenNLP provides an alternate mechanism.  You can write a `.allennlp_plugins` file, with one module
name per line, and `allennlp` will read that file and try to import all modules specified there,
just as with `--include-package`.  That file just needs to be present in whatever directory you are
running the `allennlp` command from.

</exercise>
