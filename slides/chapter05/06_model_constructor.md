---
type: slides
---

# Implementing the Model - the constructor

---

# AllenNLP Model basics

[some visualization here]

Notes: Now that we now what our model is going to do, we need to implement it.  First, we'll say a
few words about how `Models` work in AllenNLP.

An AllenNLP `Model` is just a pytorch `Module` with a particular output format (and some
other nice things we add that you don't need to worry about now).  Pytorch `Modules` implement a
`forward` method, which transform input tensors to output tensors using some parameters, providing
an easy way to train those parameters.  AllenNLP `Models` implement `forward`, and require the
output to be a _dictionary_ mapping strings to tensors, with a `loss` key provided during training.

Our training loop takes a batch of `Instances`, passes it through `Model.forward()`, grabs the
`loss` key from the resulting dictionary, and uses backprop to compute gradients and update the
model's parameters.

---

# Constructing the Model

```python
@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
```

Notes: In the `Model` constructor, we need to instantiate all of the parameters that we will want
to train.  In AllenNLP, we take most of these parameters as constructor _arguments_, so that we can
configure the behavior of our model without changing `Model.forward()`, and so that we can think at
a higher level about what our model is doing.

You'll notice that we use type annotations a lot in AllenNLP code - this is both for code
readability (it's *way* easier to understand what a method does if you know the types of its
arguments, instead of just their names), and because we _use_ these annotations to do some magic
for you in some cases.  One of those cases is constructor parameters, where we can automatically
construct the embedder and encoder from a configuration file using these type annotations.  See the
chapter on [configuration files](chapter04/) for more information.  That chapter will also tell you
about the call to `@Model.register()`.  The upshot is that if you're using the `allennlp train`
command with a configuration file, you won't ever have to call this constructor, it all gets taken
care of for you.

---

# Constructing the Model

<pre data-line="4,10"><code class="language-python">@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
</code></pre>

Notes: The vocabulary gets created by AllenNLP after reading your training data, then passed to the
`Model` when it gets constructed.  We'll find all tokens and labels that you use and assign them
all integer ids in separate namespaces.  The way that this happens is fully configurable; see the
[Vocabulary section of this course](#) for more information.  What we did in the `DatasetReader`
will put the labels in the default "labels" namespace, and we grab the number of labels from the
vocabulary on line 10.

---

# Constructing the Model

<pre data-line="5,8"><code class="language-python">@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
</code></pre>

Notes: To get an initial word embedding, we'll use AllenNLP's `TextFieldEmbedder`.  This
abstraction takes the tensors created by a `TextField` and embeds each one.  This is our most
complex abstraction, because there are a lot of ways to do this particular operation in NLP, and we
want to be able to switch between these without changing our code.  We won't go into the details
here; we have a whole [chapter of this course](#) dedicated to diving deep into how this
abstraction works and how to use it.  All you need to know for now is that you apply this to the
`text` parameter you get in `forward()`, and you get out a tensor with shape `(batch_size,
sequence_length, embedding_dim)`.

---

# Constructing the Model

<pre data-line="6,9"><code class="language-python">@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
</code></pre>

Notes: To squash our sequence of token vectors into a single vector, we use AllenNLP's
`Seq2VecEncoder` abstraction.  As the name implies, this encapsulates an operation that takes a
sequence of vectors and returns a single vector.  Because all of our modules operate on batched
input, this will take a tensor shaped like `(batch_size, sequence_length, embedding_dim)` and
return a tensor shaped like `(batch_size, encoding_dim)`.

---

# Constructing the Model

<pre data-line="11"><code class="language-python">@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
</code></pre>

Notes: The final parameters our `Model` needs is a classification layer, which can transform the
output of our `Seq2VecEncoder` into a probability distribution over all of our possible labels.  We
don't need to take this as a constructor argument, because we'll just use a simple linear
classification layer, which has sizes that we can figure out inside the constructor - the
`Seq2VecEncoder` knows its output dimension, and the `Vocabulary` knows how many labels there are.
