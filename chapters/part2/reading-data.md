---
title: 'Reading data'
description: This chapter provides a deep dive into AllenNLP abstractions that are essential for reading data, including fields and instances, dataset readers, vocabulary, and how batching is handled in AllenNLP
type: chapter
---

<exercise id="1" title="Fields and instances">

## Fields

A `Field` contains one piece of data for one example that is passed through your model. `Fields` get
converted to tensors in a model, either as an input or an output, after being converted to IDs and
batched & padded.

There are many types of fields in AllenNLP depending on the type of data they represent. Among them,
the most important is `TextFields`, which represents a piece of tokenized text. `TextFields` are so
important to AllenNLP that they get [their own chapter](/representing-text-as-features), and we
won't talk about them more here.

Other commonly used fields include:

* `LabelField` is a categorical label. We used this for representing text labels in [Part
  1](/your-first-model).
* `MultiLabelField` is an extension of `LabelField` that allows for multiple labels. This can be
  used for, e.g., multilabel text classification.
* `SequenceLabelField` is a sequence of categorical labels. This can be used for, e.g., representing
  gold labels in sequential labeling tasks.
* `SpanField` is a pair of indices that represent a span of text. This can be used for representing
  spans for reading comprehension, semantic role labeling, or coreference resolution.
* `ListField` is for when you have lists of the same type of field, e.g., multiple sentences as
  input to your model, or multiple spans.
* `ArrayField` is an array representing some data that you have already converted into a matrix,
  e.g., images and hand-crafted feature vectors.

`Fields` can be created simply by supplying the data. `Field` objects provide APIs for creating
empty fields, counting vocabulary items, creating tensors, and batching tensors, among others. See
the following code snippet for more detail.

<codeblock source="part2/reading-data/fields_source" setup="part2/reading-data/fields_setup"></codeblock>

## Instances

An instance is the atomic unit of prediction in machine learning. In AllenNLP, `Instances` are
collections of `Fields`, and datasets are collections of `Instances`.

`Instances` are created by dataset readers and (optionally) used to create a `Vocabulary`. The
`Vocabulary` is then used to map all strings in the `Instance`'s `Fields` into integer IDs, so that
they can be turned into tensors. Later in the training pipeline, these tensors are batched together
and fed to the model. The following diagram shows how `Fields` and `Instances` are created from a
dataset.

<img src="/part2/reading-data/fields-and-instances.svg" alt="Fields and Instances" />

`Instances` can be created by passing a dictionary of field names and corresponding fields to the
constructor. `Instances` know how to turn themselves into a dictionary of field names and
corresponding tensors.  The tensors for each `Field` are then batched together before being passed
to a model.  See the following code snippet for how to create instances and use their APIs.

The fields names are important—because the resulting dictionary of tensors is passed by name to the
model, they have to match the model's `forward()` arguments exactly.

<codeblock source="part2/reading-data/instances_source" setup="part2/reading-data/instances_setup"></codeblock>

</exercise>


<exercise id="2" title="Dataset readers">

## Basics of dataset readers

`DatasetReaders` read datasets and convert them to collections of `Instances`. Here, we'll give a
more in-depth look into what's going on inside a `DatasetReader`, what extra functionality it has,
and why it works the way it does.

In our [models repository](https://github.com/allenai/allennlp-models), there are dataset readers
available for a wide range of NLP tasks, including:

* [Text classification](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/text_classification_json.py)
* [Sequence labeling](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/sequence_tagging.py)
* [Language modeling](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/lm/dataset_readers/simple_language_modeling.py)
* [Natural language inference](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/pair_classification/dataset_readers/snli.py)
* [Semantic role labeling](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/srl.py)
* [Seq2Seq tasks](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/generation/dataset_readers/seq2seq.py)
* [Constituency parsing](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/penn_tree_bank.py)
  and
  [Dependency parsing](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/universal_dependencies.py)
* [Reading comprehension](https://github.com/allenai/allennlp-models/tree/master/allennlp_models/rc/dataset_readers)
* [Semantic parsing](https://github.com/allenai/allennlp-semparse)

You can implement your own dataset reader by subclassing the `DatsetReader` class. The code snippet
below is the dataset reader we implemented in [Your first model](/your-first-model). The returned
dataset is an iterable of `Instances`.

<codeblock source="part2/reading-data/dataset_reader_basic_source" setup="part2/reading-data/dataset_reader_basic_setup"></codeblock>

It is recommended that you separate out the logic for creating instances as the
`text_to_instances()` method. As we mentioned in [Training and
prediction](/training-and-prediction#4), by sharing common logic between the training and the
prediction pipelines, we are making the system less susceptible to any issues arising from possible
discrepancies in how instances are created between the two, and making it very easy to put up a demo
of your model. You can use the method as follows in, for example, your `Predictor`:

```python
reader = ClassificationTsvReader()
text = 'The best movie ever!'
label = 'pos'
instance = reader.text_to_instance(text, label)
```

`DatasetReaders` have two methods—`_read()` and `read()`. `_read()` is defined as an abstract
method, and you must override and implement your own when building a `DatasetReader` subclass for
your dataset. `read()` is the main method called by clients of the dataset reader. It basically just wraps `_read()` with some extra internal functionality. Both methods return an iterable of `Instances`.

The main method, `read()`, takes a filename as its parameter. This separates the parameterization of
the `DatasetReader` (which happens in the constructor) from the data that the reader is applied to.
This lets you apply a single `DatasetReader` to any file you want to, without having to duplicate
your parameters.

You can also design a dataset reader that handles more complex data setups. For example, you can
write one that takes a directory as its constructor parameter and takes a simple key such as `train`
and `dev` as a parameter to `read()`.  The
[`TriviaQaReader`](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/dataset_readers/triviaqa.py#L21),
for example, is designed to work this way.

Dataset readers are designed to read data from a local file, although in some cases you may want to
read data from a URL. AllenNLP provides a utility method called `cached_path` to support this. If a
URL is passed to the method it will download the resource to a local file and return its path. If
you want your dataset reader to support both local paths and URLs, you can wrap `file_path` using
`cached_path` in your `_read()` method as follows:

```python
from allennlp.common.file_utils import cached_path

...

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as lines:
            for line in lines:
```

</exercise>

<exercise id="3" title="Vocabulary">

`Vocabulary` is an important component in AllenNLP, touching on and used by many other abstractions
and components. Simply put, `Vocabulary` manages mappings from strings to integer IDs.  It is
created from instances and used for converting textual data (such as tokens and labels) to integer
IDs (and eventually to tensors).

`Vocabulary` manages different mappings using a concept called *namespaces*. Each namespace is a
distinct mapping from strings to integers, so strings in different namespaces are treated
separately. This allows you to have separate indices for, e.g., 'a' as a word and 'a' as a
character, or 'chat' in English and 'chat' in French (which means 'cat' in English). See the diagram
below for an illustration:

<img src="/part2/reading-data/vocabulary.svg" alt="Vocabulary" />

There's an important distinction between namespaces: padded and non-padded namespaces. By default,
namespaces are padded, meaning the mapping reserves indices for padding and out-of-vocabulary (OOV)
tokens. This is useful for indexing text, where OOV tokens are common and padding is needed.

Non-padded namespaces, on the other hand, do not reserve indices for special tokens. This is more
suitable for, e.g., class labels, where you don't need to worry about these. By default, namespaces
ending in `"tags"` or `"labels"` are treated as non-padded, but you can modify this behavior by
supplying a `non_padded_namespaces` parameter when creating a `Vocabulary`.

A common way to create a `Vocabulary` object is to pass a collection of `Instances` to the
`Vocabulary.from_instances()` method. This will count all strings in the `Instances` that need to be
mapped to integers, then use those counts to decide what strings should be in the vocabulary. There
are parameters to that method that customize this behavior; we'll talk about some of them below.

You can look up indices by tokens using the `get_token_index()` method. You can also do the inverse
(looking up tokens by indices) using `get_token_from_index()`.

<codeblock source="part2/reading-data/vocabulary_creation_source" setup="part2/reading-data/vocabulary_creation_setup"></codeblock>

When your vocabulary is very large, you may want to prune it by setting a threshold and only
retaining words that appear more frequently than that threshold. You can achieve this by passing a
`min_count` parameter to `Vocabulary.from_instances()`, which specifies the minimum count tokens
need to meet to be included per namespace.

<codeblock source="part2/reading-data/vocabulary_count_source" setup="part2/reading-data/vocabulary_count_setup"></codeblock>

You can instantiate `Vocabulary` not just from a collection of instances but by other means. The
class method `from_files` allows you to load a serialized `Vocabulary` from a directory. This is
often one created by a previous run of `allennlp train`. You can also use `from_files_and_instances`
to expand a pre-built vocabulary with new data.  `Vocabulary` is registered using all of these
constructors, so you can create a vocabulary from a configuration file using any of these methods.
`from_instances` is the default constructor to use; `"type": "from_files"` will use the `from_files`
method, while `"type": "extend"` will use `from_files_and_instances`.

Unless you are writing your own training script, you rarely need to worry about how `Vocabulary` is
built and managed. You never "see" the vocabulary in your dataset reader—it will be constructed
behind the scenes by AllenNLP and used by the `DataLoader` to index the instances. If you're using a
pretrained contextualizer, its pre-built vocabulary is typically added automatically for you.

The final constructed vocab gets passed to the model automatically. In the model constructor, you
can use the information from `Vocabulary` to initialize model parameters. For example, in the
`SimpleClassifier` model we built in Part 1, the size of the `"labels"` namespace is used to
initialize the final linear layer of the classifier:

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
        self.accuracy = CategoricalAccuracy()
</code></pre>

</exercise>


<exercise id="4" title="Datasets, the dataset loader, and samplers">

AllenNLP turns `Instance` objects into batches of tensors through a [`DataLoader`](http://docs.allennlp.org/main/api/data/data_loaders/data_loader/#dataloader).

For most scenarios, the [`MultiProcessDataLoader`](http://docs.allennlp.org/main/api/data/data_loaders/multiprocess_data_loader/#multiprocessdataloader) is the implementation you'll want to use. This `DataLoader` takes a `DatasetReader` along with a `path` to feed to the `DatasetReader` and produces an iterable of batched tensor dictionaries.

In the simplest use case, you can just supply the `batch_size` parameter and it will batch instances together in groups of `batch_size` in the order that it gets them from the `DatasetReader`.

<codeblock source="part2/reading-data/data_loader_basic" setup="part2/reading-data/data_loader_setup"></codeblock>

You can easily customize how the `DataLoader` iterates over, samples, and/or batches the instances by specifying a [`BatchSampler`](http://docs.allennlp.org/main/api/data/samplers/batch_sampler/#batchsampler).

The [`BucketBatchSampler`](http://docs.allennlp.org/main/api/data/samplers/bucket_batch_sampler/#bucketbatchsampler) is the most important `BatchSampler` in practice and is a major feature of AllenNLP.
It sorts the instances by the length of their longest `Field` (or by any sorting keys you specify)
and automatically groups them so that instances of similar lengths get batched together. This helps
minimize the amount of padding and makes training more efficient. `Fields` all implement `__len__`,
which the sampler uses to do this sorting, and to figure out which `Field` to sort by.  For example,
if you have two `TextFields`, one for a question and one for a passage, the `BucketBatchSampler`
will automatically detect that it should sort by passage length. In the following code example, we
compare `BasicBatchSampler` and `BucketBatchSampler`; notice the difference in the amount of padding
between the two.

<codeblock source="part2/reading-data/data_loader_bucket" setup="part2/reading-data/data_loader_setup"></codeblock>

</exercise>
