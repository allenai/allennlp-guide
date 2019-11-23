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

<img src="/your-first-model/dataset-reader.png" alt="How dataset reader works" />

The first step for building an NLP application is to read the dataset and represent it with some internal data structure. 

AllenNLP uses `DatasetReader`s to read the data, whose job it is to transform raw data files into [`Instances`](/reading-textual-data) that match the input / output spec. Our spec for text classification is:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

We'll want one [`Field`](/reading-textual-data) for the input and another for the output, and our model will use the inputs to predict the outputs.

We assume the dataset has a simple data fle format:
`[text] [TAB] [label]`, for example:

```
I like this movie a lot! [TAB] positive
This was a monstrous waste of time [TAB] negative
AllenNLP is amazing [TAB] positive
Why does this have to be some complicated? [TAB] negative
This sentence expresses no sentiment [TAB] neutral
```

</exercise>

<exercise id="2" title="Make a DatasetReader">

You can implement your own `DatasetReader` by inheriting the `DatasetReader` class. At minimum, you need to override the `_read()` method, which reads the input dataset and yields `Instance`s.

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
</exercise>

<exercise id="4" title="Writing a config file">

<codeblock id="your-first-model/config">
Try changing the configuration parameters and see how the dataset reader and model change.  In
particular, see if you can add a character-level CNN to the `TextField` parameters.  You'll need to
add parameters both for the `DatasetReader` (inside a `token_indexers` block) and for the
`Model` (inside the `embedder` block).
</codeblock>

</exercise>
