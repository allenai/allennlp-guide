---
type: slides
---

# Reading Data

---

# What we need to do

Some figure showing `[input text] --> [dataset reader] --> [instances matching input/output spec]`

Notes: To read data we use a `DatasetReader`, whose job it is to transform raw data files into
[`Instances`](#) that match our input / output spec.  We'll want one [`Field`](#) per item in our
spec, and our model will use the inputs to predict the outputs.

Ideally, the outputs would be optional in the `Instances`, so that we can use the same code to make
predictions on unlabeled data (say, in a demo), but for the rest of this chapter we'll keep things
simple and ignore that.

---

# Get some data

Our input / output spec is:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

So we need data that has text and a label.  Let's assume a simple data file format: `[text] [TAB]
[label]`.

```
I like this movie a lot! [TAB] positive
This was a monstrous waste of time [TAB] negative
AllenNLP is amazing [TAB] positive
Why does this have to be some complicated? [TAB] negative
This sentence expresses no sentiment [TAB] neutral
```

---

# Make a DatasetReader

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

Notes: This is a minimal `DatasetReader` that will return a list of classification `Instances` when
you call `reader.read(file)`.  This reader will take each line in the input file, split the `text`
into words using the default word tokenizer (the default currently relies on
[spacy](https://spacy.io/)'s tokenizer), and represent those words as tensors using a word id in a
vocabulary we construct for you.

Pay special attention to the `text` and `label` keys that were used in the `fields` dictionary
passed to the `Instance` - these keys will be used as parameter names when passing tensors into
your `Model` later.

There are lots of places where this could be made better for a more flexible and fully-featured
reader; see the section on [DatasetReaders](#) for more information.
