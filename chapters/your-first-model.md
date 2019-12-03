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

## AllenNLP Model basics

<img src="/your-first-model/allennlp-model.svg" alt="AllenNLP model" />

Now that we now what our model is going to do, we need to implement it. First, we'll say a few words about how `Models` work in AllenNLP:

* AllenNLP `Model` is just a PyTorch `Module`
* It implements a `forward()` method, and requires the output to be a _dictionary_
* Its output contains a `loss` key during training, which is used to optimize the model

Our training loop takes a batch of `Instances`, passes it through `Model.forward()`, grabs the `loss` key from the resulting dictionary, and uses backprop to compute gradients and update the model's parameters.

</exercise>

<exercise id="4" title="Implementing the model — the constructor" type="slides">

<slides source="your-first-model/implementing-model-constructor" />

</exercise>

<exercise id="5" title="Writing a config file">

<codeblock id="your-first-model/config">
Try changing the configuration parameters and see how the dataset reader and model change.  In
particular, see if you can add a character-level CNN to the `TextField` parameters.  You'll need to
add parameters both for the `DatasetReader` (inside a `token_indexers` block) and for the
`Model` (inside the `embedder` block).
</codeblock>

</exercise>
