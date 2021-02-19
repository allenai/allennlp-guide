---
title: 'Your first model'
description:
  "In this chapter you are going to build your first text classification model using AllenNLP."
type: chapter
---

<textblock>

In this section of the guide, we'll give a quick start on one the most basic things you can do with
AllenNLP: text classification.  We give a brief introduction to text classification, then implement
a simple classifier that decides whether a movie review expresses positive or negative sentiment.

</textblock>

<exercise id="1" title="What is text classification">

Text classification is one of the simplest NLP tasks, where the model, given some input text,
predicts a label for the text. See the figure below for an illustration.

<img src="/part1/introduction/text-classification.svg" alt="Text classification" />

There are a variety of applications of text classification, such as spam filtering, sentiment
analysis, and topic detection.  Some examples are shown in the table below.

| Application        | Description                   | Input                      | Output                   |
|--------------------|-------------------------------|----------------------------|--------------------------|
| Spam filtering     | Detect and filter spam emails | Email                      | Spam / Not spam          |
| Sentiment analysis | Detect the polarity of text   | Tweet, review              | Positive / Negative      |
| Topic detection    | Detect the topic of text      | News article, blog post    | Business / Tech / Sports |

</exercise>


<exercise id="2" title="Defining input and output">

The first step for building an NLP model is to define its input and output. In AllenNLP, each
training example is represented by an `Instance` object. An `Instance` consists of one or more
`Fields`, where each `Field` represents one piece of data used by your model, either as an input or
an output. `Fields` will get converted to tensors and fed to your model. The [Reading Data
chapter](/reading-data) provides more details on using `Instances` and `Fields` to represent textual
data.

For text classification, the input and the output are very simple. The model takes a `TextField`
that represents the input text and predicts its label, which is represented by a `LabelField`:

```python
# Input
text: TextField

# Output
label: LabelField
```

</exercise>


<exercise id="3" title="Reading data">

<img src="/part1/your-first-model/dataset-reader.svg" alt="How dataset reader works" />

The first step for building an NLP application is to read the dataset and represent it with some
internal data structure.

AllenNLP uses `DatasetReaders` to read the data, whose job it is to transform raw data files into
`Instances` that match the input / output spec. Our spec for text classification is:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

We'll want one `Field` for the input and another for the output, and our model will use the inputs
to predict the outputs.

We assume the dataset has a simple data file format:
`[text] [TAB] [label]`, for example:

```
I like this movie a lot! [TAB] positive
This was a monstrous waste of time [TAB] negative
AllenNLP is amazing [TAB] positive
Why does this have to be so complicated? [TAB] negative
This sentence expresses no sentiment [TAB] neutral
```

</exercise>


<exercise id="4" title="Making a DatasetReader">

You can implement your own `DatasetReader` by inheriting from the `DatasetReader` class. At minimum,
you need to override the `_read()` method, which reads the input dataset and yields `Instances`.

```python
@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self):
        self.tokenizer = SpacyTokenizer()
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, label = line.strip().split('\t')
                text_field = TextField(self.tokenizer.tokenize(text),
                                       self.token_indexers)
                label_field = LabelField(label)
                fields = {'text': text_field, 'label': label_field}
                yield Instance(fields)
```

This is a minimal `DatasetReader` that will return a list of classification `Instances` when you
call `reader.read(file)`.  This reader will take each line in the input file, split the `text` into
words using a tokenizer (the `SpacyTokenizer` shown here relies on [spaCy](https://spacy.io/)), and
represent those words as tensors using a word id in a vocabulary we construct for you.

Pay special attention to the `text` and `label` keys that are used in the `fields` dictionary passed
to the `Instance` - these keys will be used as parameter names when passing tensors into your
`Model` later.

Ideally, the output label would be optional when we create the `Instances`, so that we can use the
same code to make predictions on unlabeled data (say, in a demo), but for the rest of this chapter
we'll keep things simple and ignore that.

There are lots of places where this could be made better for a more flexible and fully-featured
reader; see the section on [DatasetReaders](/reading-data#2) for a deeper dive.

</exercise>


<exercise id="5" title="Building your model">

<img src="/part1/your-first-model/designing-a-model.svg" alt="Designing a model" />

The next thing we need is a `Model` that will take a batch of `Instances`, predict the outputs from
the inputs, and compute a loss.

Remember that our `Instances` have this input/output spec:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

Also, remember that we *used these names* (`text` and `label`) for the fields in the
`DatasetReader`. AllenNLP passes those fields by name to the model code, so we need to use the same
names in our model.

## What should our model do?

<img src="/part1/your-first-model/designing-a-model-1.svg" alt="Designing a model 1" />

Conceptually, a generic model for classifying text does the following:

* Get some features corresponding to each word in your input
* Combine those word-level features into a document-level feature vector
* Classify that document-level feature vector into one of your labels.

In AllenNLP, we make each of these conceptual steps into a generic abstraction that you can use in
your code, so that you can have a very flexible model that can use different concrete components for
each step.

## Representing text with token IDs

<img src="/part1/your-first-model/designing-a-model-2.svg" alt="Designing a model 2" />

The first step is changing the strings in the input text into token ids.  This is handled by the
`SingleIdTokenIndexer` that we used previously, during part of our data processing pipeline that you
don't have to write code for.

## Embedding tokens

<img src="/part1/your-first-model/designing-a-model-3.svg" alt="Designing a model 3" />

The first thing our `Model` does is apply an `Embedding` function that converts each token ID that
we got as input into a vector.  This gives us a vector for each input token, so we have a large
tensor here.

## Apply Seq2Vec encoder

<img src="/part1/your-first-model/designing-a-model-4.svg" alt="Designing a model 4" />

Next we apply some function that takes the sequence of vectors for each input token and squashes it
into a single vector. Before the days of pretrained language models like BERT, this was typically an
LSTM or convolutional encoder.  With BERT we might just take the embedding of the `[CLS]` token
(more on how to do that [later](/next-steps)).

## Computing distribution over labels

<img src="/part1/your-first-model/designing-a-model-5.svg" alt="Designing a model 5" />

Finally, we take that single feature vector (for each `Instance` in the batch), and classify it as a
label, which will give us a categorical probability distribution over our label space.

</exercise>


<exercise id="6" title="Implementing the model — the constructor">

## AllenNLP Model basics

<img src="/part1/your-first-model/allennlp-model.svg" alt="AllenNLP model" />

Now that we know what our model is going to do, we need to implement it. First, we'll say a few words
about how `Models` work in AllenNLP:

* An AllenNLP `Model` is just a PyTorch `Module`
* It implements a `forward()` method, and requires the output to be a _dictionary_
* Its output contains a `loss` key during training, which is used to optimize the model

Our training loop takes a batch of `Instances`, passes it through `Model.forward()`, grabs the
`loss` key from the resulting dictionary, and uses backprop to compute gradients and update the
model's parameters. You don't have to implement the training loop—all this will be taken care of by
AllenNLP (though you can if you want to).

## Constructing the Model

In the `Model` constructor, we need to instantiate all of the parameters that we will want to train.
In AllenNLP, [we recommend](/using-config-files#1) taking most of these parameters as constructor
_arguments_, so that we can configure the behavior of our model without changing the model code
itself, and so that we can think at a higher level about what our model is doing. The constructor
for our text classification model looks like this:

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

You'll notice that we use type annotations a lot in AllenNLP code - this is both for code
readability (it's *way* easier to understand what a method does if you know the types of its
arguments, instead of just their names), and because we _use_ these annotations to do some magic
for you in some cases.

One of those cases is constructor parameters, where we can automatically construct the embedder and
encoder from a configuration file using these type annotations. See the chapter on [configuration
files](/using-config-files) for more information. That chapter will also tell you about the call to
`@Model.register()`.

The upshot is that if you're using the `allennlp train` command with a configuration file (which we
show how to do below), you won't ever have to call this constructor, it all gets taken care of for
you.

### Passing the vocabulary

<pre data-line="4,10" class="language-python line-numbers"><code>
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
</code></pre>

`Vocabulary` manages mappings between vocabulary items (such as words and labels) and their integer
IDs. In our prebuilt training loop, the vocabulary gets created by AllenNLP after reading your
training data, then passed to the `Model` when it gets constructed. We'll find all tokens and labels
that you use and assign them all integer IDs in separate namespaces. The way that this happens is
fully configurable; see the [Vocabulary section of this guide](/reading-data#3) for more
information.

What we did in the `DatasetReader` will put the labels in the default "labels" namespace, and we
grab the number of labels from the vocabulary on line 10.

### Embedding words

<pre data-line="5,8" class="language-python line-numbers"><code>
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
</code></pre>

To get an initial word embedding, we'll use AllenNLP's `TextFieldEmbedder`. This abstraction takes
the tensors created by a `TextField` and embeds each one. This is our most complex abstraction,
because there are a lot of ways to do this particular operation in NLP, and we want to be able to
switch between these without changing our code.  We won't go into the details here; we have a whole
[chapter of this guide](/representing-text-as-features) dedicated to diving deep into how this
abstraction works and how to use it. All you need to know for now is that you apply this to the
`text` parameter you get in `forward()`, and you get out a tensor that has a single embedding vector
for each input token, with shape `(batch_size, num_tokens, embedding_dim)`.

### Applying a Seq2VecEncoder

<pre data-line="6,9" class="language-python line-numbers"><code>
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
</code></pre>

To squash our sequence of token vectors into a single vector, we use AllenNLP's `Seq2VecEncoder`
abstraction. As the name implies, this encapsulates an operation that takes a sequence of vectors
and returns a single vector. Because all of our modules operate on batched input, this will take a
tensor shaped like `(batch_size, num_tokens, embedding_dim)` and return a tensor shaped like
`(batch_size, encoding_dim)`.

### Applying a classification layer

<pre data-line="11" class="language-python line-numbers"><code>
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
</code></pre>

The final parameters our `Model` needs is a classification layer, which can transform the output of
our `Seq2VecEncoder` into logits, one value per possible label. These values will be converted to a
probability distribution later and used for calculating the loss.

We don't need to take this as a constructor argument, because we'll just use a simple linear layer,
which has sizes that we can figure out inside the constructor - the `Seq2VecEncoder` knows its
output dimension, and the `Vocabulary` knows how many labels there are.

</exercise>


<exercise id="7" title="Implementing the model — the forward method">

Next, we need to implement the `forward()` method of your model, which takes the input, produces the
prediction, and computes the loss. Remember, our constructor and input/output spec look like:

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

```python
# Inputs:
text: TextField

# Outputs:
label: LabelField
```

Here we'll show how to use these parameters inside of `Model.forward()`, which will get arguments
that match our input/output spec (because that's how we coded the
[`DatasetReader`](/your-first-model#4)).

## Model.forward()

In `forward`, we use the parameters that we created in our constructor to transform the inputs into
outputs. After we've predicted the outputs, we compute some loss function based on how close we got
to the true outputs, and then return that loss (along with whatever else we want) so that we can use
it to train the parameters.

```python
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
```

### Inputs to forward()

<pre data-line="3-4" class="language-python line-numbers"><code>
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

The first thing to notice is the inputs to this function. The way the AllenNLP training loop works
is that we will take the field names that you used in your `DatasetReader` and give you a batch of
instances with _those same field names_ in `forward`. So, because we used `text` and `label` as our
field names, we need to name our arguments to `forward` the same way.

Second, notice the types of these arguments. Each type of `Field` knows how to convert itself into a
`torch.Tensor`, then create a batched `torch.Tensor` from all of the `Fields` with the same name
from a batch of `Instances`. The types you see for `text` and `label` are the tensors produced by
`TextField` and `LabelField` (again, see our [chapter on using
TextFields](/representing-text-as-features) for more information about `TextFieldTensors`). The
important part to know is that our `TextFieldEmbedder`, which we created in the constructor, expects
this type of object as input and will return an embedded tensor as output.

### Embedding the text

<pre data-line="5-6" class="language-python line-numbers"><code>
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

The first actual modeling operation that we do is embed the text, getting a vector for each input
token. Notice here that we're not specifying anything about _how_ that operation is done, just that
a `TextFieldEmbedder` that we got in our constructor is going to do it. This lets us be very
flexible later, changing between various kinds of embedding methods or pretrained representations
(including ELMo and BERT) without changing our model code.

### Applying a Seq2VecEncoder

<pre data-line="7-10" class="language-python line-numbers"><code>
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

After we have embedded our text, we next have to squash the sequence of vectors (one per token) into
a single vector for the whole text. We do that using the `Seq2VecEncoder` that we got as a
constructor argument. In order to behave properly when we're batching pieces of text together that
could have different lengths, we need to _mask_ elements in the `embedded_text` tensor that are only
there due to padding.  We use a utility function to get a mask from the `TextField` output, then
pass that mask into the encoder.

At the end of these lines, we have a single vector for each instance in the batch.

### Making predictions

<pre data-line="11-16" class="language-python line-numbers"><code>
class SimpleClassifier(Model):
    def forward(self,
                text: TextFieldTensors,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

The last step of our model is to take the vector for each instance in the batch and predict a label
for it. Our `classifier` is a `torch.nn.Linear` layer that gives a score (commonly called a `logit`)
for each possible label. We normalize those scores using a `softmax` operation to get a probability
distribution over labels that we can return to a consumer of this model. For computing the loss,
PyTorch has a built in function that computes the cross entropy between the logits that we predict
and the true label distribution, and we use that as our loss function.

And that's it! This is all you need for a simple classifier.  After you've written a `DatasetReader`
and `Model`, AllenNLP takes care of the rest: connecting your input files to the dataset reader,
intelligently batching together your instances and feeding them to the model, and optimizing the
model's parameters by using backprop on the loss.  We go over this part in the next chapter.

</exercise>
