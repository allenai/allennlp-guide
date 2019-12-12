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

* An AllenNLP `Model` is just a PyTorch `Module`
* It implements a `forward()` method, and requires the output to be a _dictionary_
* Its output contains a `loss` key during training, which is used to optimize the model

Our training loop takes a batch of `Instances`, passes it through `Model.forward()`, grabs the `loss` key from the resulting dictionary, and uses backprop to compute gradients and update the model's parameters. You don't have to implement the training loop—all this will be taken care of by AllenNLP.

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

The upshot is that if you're using the `allennlp train` command with a configuration file (as we'll do below), you won't ever have to call this constructor, it all gets taken care of for you.

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

`Vocabulary` manages mappings between vocabulary items (such as words and characters) and their integer IDs. In our prebuilt training loop, the vocabulary gets created by AllenNLP after reading your training data, then passed to the `Model` when it gets constructed. We'll find all tokens and labels that you use and assign them all integer IDs in separate namespaces. The way that this happens is fully configurable; see the [Vocabulary section of this course](/reading-textual-data) for more information.

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

<exercise id="5" title="Implementing the model — the forward method">

Next, we need to implement the `forward()` method of your model, which takes the input, produces the prediction, and computes the loss. Remember, our constructor and input/output spec look like:

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

Here we'll show how to use these parameters inside of `Model.forward()`, which will get arguments that match our input/output spec (because that's how we coded the [`DatasetReader`](/your-first-model#2)).

## Model.forward()

In `forward`, we use the parameters that we created in our constructor to transform the inputs into outputs. After we've predicted the outputs, we compute some loss function based on how close we got to the true outputs, and then return that loss (along with whatever else we want) so that we can use it to train the parameters.

```python
class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
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

## Inputs to forward()

<pre data-line="4-5" class="language-python"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
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

The first thing to notice is the inputs to this function. The way the AllenNLP training loop works is that we will take the field names that you used in your `DatasetReader` and give you a batch of instances with _those same field names_ in `forward`. So, because we used `text` and `label` as our field names, we need to name our arguments to `forward` the same way.

Second, notice the types of these arguments. Each type of `Field` knows how to convert itself into a `torch.Tensor`, then create a batched `torch.Tensor` from all of the `Fields` with the same name from a batch of `Instances`. The types you see for `text` and `label` are the tensors produced by `TextField` and `LabelField`. We won't go into the details of why `TextField` produces a `Dict[str, torch.Tensor]` here; see our [chapter on using TextFields](/representing-text-as-features) for more information about that. The important part to know is that our `TextFieldEmbedder`, which we created in the constructor, expects this type of object as input and will return an embedded tensor as output.

## Embedding the text

<pre data-line="6-7" class="language-python"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
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

The first actual modeling operation that we do is embed the text, getting a vector for each input token. Notice here that we're not specifying anything about _how_ that operation is done, just that a `TextFieldEmbedder` that we got in our constructor is going to do it. This lets us be very flexible later, changing between various different kinds of embedding methods or pretrained representations (including ELMo and BERT) without changing our code.

## Applying a Seq2VecEncoder

<pre data-line="8-11" class="language-python"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
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

After we have embedded our text, we next have to squash the sequence of vectors (one per token) into a single vector for the whole text. We do that using the `Seq2VecEncoder` that we got as a constructor argument. In order to behave properly when we're batching pieces of text together that could have different lengths, we need to _mask_ elements in the `embedded_text` tensor that are only there due to padding.  We use a utility function to get a mask from the `TextField` output, then pass that mask into the encoder. Our [deep dive on building models](/building-your-model) will give you more details on padding and masking.

At the end of these lines, we have a single vector for each instance in the batch.

## Making predictions

<pre data-line="12-17" class="language-python"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
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

The last step of our model is to take the vector for each instance in the batch and predict a label for it. Our `classifier` is a `torch.nn.Linear` layer that gives a score (commonly called a `logit`) for each possible label. We normalize those scores using softmax to get a probability distribution over labels that we can return to a consumer of this model. For computing the loss, PyTorch has a built in function that computes the cross entropy between the logits that we predict and the true label distribution, and we use that as our loss function.

And that's it! This is all you need for a simple classifier.

</exercise>

<exercise id="6" title="Writing a config file">

As mentioned previously, in AllenNLP you don't need to worry about connecting the input files to the dataset reader, batching, feeding the data to the model, or writing the training loop. You just need to specify how individual components (such as the dataset reader, the model, the optimizer etc.) get initialized along with their parameters by writing *a configuration file*. (Well, you can do these things if you want to, but we think it's easier to use the configuration files in most cases.)

Config files in AllenNLP are formatted in JSON (or more specifically, a superset of JSON called [Jsonnet](https://jsonnet.org/), which supports fancier features like variables and imports). A config file is just a big JSON object with several keys (or *sections*) corresponding to individual components of your project:

```js
{
    "dataset_reader": [... config for the dataset reader ...],
    "train_data_path": [... path to the training data ...],
    "validation_data_path": [... path to the validation data ...],
    "model": [... config for the model ...],
    "iterator": [... config for the iterator ...],
    "trainer": [... config for the trainer ...]
}
```

## Initializing the dataset reader

Usually each JSON object in a config file corresponds to a Python object. For example, in the first section, you specify how the dataset reader should be initialized:

```js
"dataset_reader" : {
    "type": "classification-tsv",
    "token_indexers": {
        "tokens": {
            "type": "single_id"
        }
    }
}
```

The first key, `type`, tells which subclass of `DatasetReader` to use. Most AllenNLP classes inherit from the `Registrable` class, which allows you to refer to a subclass by its registered name. Because earlier you did:

<pre data-line="2" class="language-python"><code class="language-python">@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
    ...
</code></pre>

when you defined your dataset reader, you can use its name `classification-tsv` in the config file. 

Other keys in a config JSON object correspond to constructor parameters. Here we are telling the dataset reader to use a dictionary for `token_indexers`, which has a single key `tokens` in it. The value of `tokens` is again a JSON object whose `type` is `single_id`, meaning a `SingleIdTokenIndexer` will be used. This process gets repeated recursively as needed. AllenNLP looks at the type annotation on each constructor parameter, and tries to build an object of that type from the corresponding parameters in the JSON file.

In summary, the config section for the dataset reader above has the same effect as the following Python snippet:

```python
reader = ClassificationTsvReader(
    token_indexers={'tokens': SingleIdTokenIndexer()})
```

## Initializing the model

The section for the model works in a very similar way as the one for the dataset reader. Here's a sample config section for initializing the simple classifier we implemented:

```javascript
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
}
```

As with dataset readers, AllenNLP models inherit from `Registrable`, which allows you to refer to model subclasses by their registered names. Remember that earlier we did:

<pre data-line="2" class="language-python"><code class="language-python">@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        ...
</code></pre>

The model section instantiates a `simple_classifier` (which is the `SimpleClassifier` class you just defined) with the specified constructor parameters (`embedder` and `encoder`). We are not going into the details here, but the model section above has the same effect as:

```python
model = SimpleClassifier(
    embedder=BasicTextFieldEmbedder(
        token_embedders={
            'tokens': Embedding(
                num_embeddings=vocab.get_vocab_size('tokens'),
                embedding_dim=10)}),
    encoder=BagOfEmbeddingsEncoder(
        embedding_dim=10))
```

Note that `vocab` is automatically handled by AllenNLP, so you don't need to pass it around explicitly. If you are interested in learning more about how config files work, see the [chapter on using config files](/using-config-files) in Part 3.

In the next chapter, we'll start training and making predictions using our text classification model!

</exercise>
