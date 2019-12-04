---
title: 'Your first model'
description:
  "In this chapter you are going to build your first text classification model using AllenNLP."
prev: /introduction
next: /training-and-prediction
type: chapter
id: 202
---

<textblock>

In the previous chapter, we learned what text classification is. In this part of the course, we are going to build a simple text classification model using AllenNLP which classifies movie reviews based on their polarity (that is, positive or negative). 

</textblock>

<exercise id="1" title="Reading data">

<img src="/your-first-model/dataset-reader.svg" alt="How dataset reader works" />

The first step for building an NLP application is to read the dataset and represent it with some internal data structure. 

AllenNLP uses `DatasetReaders` to read the data, whose job it is to transform raw data files into [`Instances`](/reading-textual-data) that match the input / output spec. Our spec for text classification is:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

We'll want one [`Field`](/reading-textual-data) for the input and another for the output, and our model will use the inputs to predict the outputs.

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

<exercise id="2" title="Make a DatasetReader">

You can implement your own `DatasetReader` by inheriting from the `DatasetReader` class. At minimum, you need to override the `_read()` method, which reads the input dataset and yields `Instances`.

```python
@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self):
        self.tokenizer = WordTokenizer()
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

This is a minimal `DatasetReader` that will return a list of classification `Instances` when you call `reader.read(file)`.  This reader will take each line in the input file, split the `text` into words using the default word tokenizer (the default currently relies on [spaCy](https://spacy.io/)'s tokenizer), and represent those words as tensors using a word id in a vocabulary we construct for you.

Pay special attention to the `text` and `label` keys that were used in the `fields` dictionary passed to the `Instance` - these keys will be used as parameter names when passing tensors into your `Model` later.

Ideally, the outputs would be optional in the `Instances`, so that we can use the same code to make predictions on unlabeled data (say, in a demo), but for the rest of this chapter we'll keep things simple and ignore that.

There are lots of places where this could be made better for a more flexible and fully-featured reader; see the section on [DatasetReaders](/reading-textual-data) for more information.

</exercise>

<exercise id="3" title="Building your model">

<img src="/your-first-model/designing-a-model.svg" alt="Designing a model" />

The next thing we need is a `Model` that will take a batch of `Instances`, predict the outputs from the inputs, and compute a loss.

After you've written `DatasetReader` and `Model`, AllenNLP takes care of the rest: connecting your input files to the dataset reader, intelligently batching together your instances and feeding them to the model, and optimizing the model's parameters by using backprop on the loss.

Remember that our `Instances` have this input/output spec:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

Also, remember that we *used these names* (`text` and `label`) for the fields in the `DatasetReader`. We need to use the same names in our model.

## What should our model do?

<img src="/your-first-model/designing-a-model-1.svg" alt="Designing a model 1" />

Conceptually, a generic model for classifying text does the following:

* Get some features corresponding to each word in your input
* Combine those word-level features into a document-level feature vector
* Classify that document-level feature vector into one of your labels.

In AllenNLP, we make each of these conceptual steps into a generic abstraction that you can use in your code, so that you can have a very flexible model that can use different concrete components for each step, just by changing a configuration file.

## Representing text with token IDs

<img src="/your-first-model/designing-a-model-2.svg" alt="Designing a model 2" />

The first step is changing the strings in the input text into token ids, which is handled by the `DatasetReader` (that's done by the `SingleIdTokenIndexer` that we used previously).

## Embedding tokens

<img src="/your-first-model/designing-a-model-3.svg" alt="Designing a model 3" />

The first thing our `Model` does is apply an `Embedding` function that converts each token ID that we got as input into a vector.  This gives us a vector for each input token, so we have a large tensor here.

## Apply Seq2Vec encoder

<img src="/your-first-model/designing-a-model-4.svg" alt="Designing a model 4" />

Next we apply some function that takes the sequence of vectors for each input token and
squashes it into a single vector. Before the days of pretrained language models like BERT, this was typically an LSTM or convolutional encoder.  With BERT we might just take the embedding of the `[CLS]` token (more on [how to do that later](/next-steps)).

## Computing distribution over labels

<img src="/your-first-model/designing-a-model-5.svg" alt="Designing a model 5" />

Finally, we take that single feature vector (for each `Instance` in the batch), and classify it as a label, which will give us a categorical probability distribution over our label space.

</exercise>

<exercise id="4" title="Implementing the model — the constructor">

## AllenNLP Model basics

<img src="/your-first-model/allennlp-model.svg" alt="AllenNLP model" />

Now that we now what our model is going to do, we need to implement it. First, we'll say a few words about how `Models` work in AllenNLP:

* AllenNLP `Model` is just a PyTorch `Module`
* It implements a `forward()` method, and requires the output to be a _dictionary_
* Its output contains a `loss` key during training, which is used to optimize the model

Our training loop takes a batch of `Instances`, passes it through `Model.forward()`, grabs the `loss` key from the resulting dictionary, and uses backprop to compute gradients and update the model's parameters.

# Constructing the Model

In the `Model` constructor, we need to instantiate all of the parameters that we will want to train. In AllenNLP, we take most of these parameters as constructor _arguments_, so that we can configure the behavior of our model without changing `Model.forward()`, and so that we can think at a higher level about what our model is doing. The constructor for our text classification model looks like:

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

You'll notice that we use type annotations a lot in AllenNLP code - this is both for code readability (it's *way* easier to understand what a method does if you know the types of its arguments, instead of just their names), and because we _use_ these annotations to do some magic for you in some cases.

One of those cases is constructor parameters, where we can automatically construct the embedder and encoder from a configuration file using these type annotations. See the chapter on [configuration files](#) for more information. That chapter will also tell you about the call to `@Model.register()`.

The upshot is that if you're using the `allennlp train` command with a configuration file, you won't ever have to call this constructor, it all gets taken care of for you.

## Passing the vocabulary

<pre data-line="5,11" class="language-python"><code class="language-python">@Model.register('simple_classifier')
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

`Vocabulary` manages mappings between vocabulary items (such as words and characters) and their integer IDs. The vocabulary gets created by AllenNLP after reading your training data, then passed to the `Model` when it gets constructed. We'll find all tokens and labels that you use and assign them all integer IDs in separate namespaces. The way that this happens is fully configurable; see the [Vocabulary section of this course](/reading-textual-data) for more information.

What we did in the `DatasetReader` will put the labels in the default "labels" namespace, and we grab the number of labels from the vocabulary on line 11.

## Embedding words

<pre data-line="6,9" class="language-python"><code class="language-python">@Model.register('simple_classifier')
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

To get an initial word embedding, we'll use AllenNLP's `TextFieldEmbedder`. This
abstraction takes the tensors created by a `TextField` and embeds each one. This is our most complex abstraction, because there are a lot of ways to do this particular operation in NLP, and we want to be able to switch between these without changing our code.  We won't go into the details here; we have a whole [chapter of this course](/representing-text-as-features) dedicated to diving deep into how this abstraction works and how to use it. All you need to know for now is that you apply this to the `text` parameter you get in `forward()`, and you get out a tensor with shape `(batch_size, num_tokens, embedding_dim)`.

## Applying a Seq2VecEncoder

<pre data-line="7,10" class="language-python"><code class="language-python">@Model.register('simple_classifier')
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

To squash our sequence of token vectors into a single vector, we use AllenNLP's
`Seq2VecEncoder` abstraction. As the name implies, this encapsulates an operation that takes a sequence of vectors and returns a single vector. Because all of our modules operate on batched input, this will take a tensor shaped like `(batch_size, num_tokens, embedding_dim)` and return a tensor shaped like `(batch_size, encoding_dim)`.

## Applying a classification layer

<pre data-line="12" class="language-python"><code class="language-python">@Model.register('simple_classifier')
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

The final parameters our `Model` needs is a classification layer, which can transform the output of our `Seq2VecEncoder` into logits, one value per possible label. These values will be converted to a probability distribution later and used for calculating the loss.

We don't need to take this as a constructor argument, because we'll just use a simple linear layer, which has sizes that we can figure out inside the constructor - the `Seq2VecEncoder` knows its output dimension, and the `Vocabulary` knows how many labels there are.

</exercise>

<exercise id="5" title="Writing a config file">

<codeblock id="your-first-model/config">
Try changing the configuration parameters and see how the dataset reader and model change.  In
particular, see if you can add a character-level CNN to the `TextField` parameters.  You'll need to
add parameters both for the `DatasetReader` (inside a `token_indexers` block) and for the
`Model` (inside the `embedder` block).
</codeblock>

</exercise>
