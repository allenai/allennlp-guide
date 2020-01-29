---
title: 'Reading textual data'
description: This chapter provides a deep dive into AllenNLP abstractions that are essential for reading textual data, including fields and instances, dataset readers, vocabulary, and how batching is handled in AllenNLP
prev: /next-steps
next: /building-your-model
type: chapter
id: 301
---

<exercise id="1" title="Fields and instances">

## Fields

A `Field` contains one piece of data for one example that is passed through your model. `Fields` get converted to tensors in a model, either as an input or an output, after being converted to IDs and batched & padded. 

There are many types of fields in AllenNLP depending on the type of data they represent. Among them, the most important is `TextFields`, which represents a piece of tokenized text. [A chapter on representing text as features](/representing-text-as-features) will give you a deep dive into `TextFields` and related concepts.

Other commonly used fields include:

* `LabelField` is a categorical label. We used this for representing text labels in [Part 1 — quick start](/your-first-model).
* `MultiLabelField` is an extension of `LabelField` that allows for multiple labels. This can be used for, e.g., multilabel text classification.
* `SequenceLabelField` is a sequence of categorical labels. This can be used for, e.g., representing gold labels in sequential labeling tasks.
* `SpanField` is a pair of indices that represent a span of text. This can be used for, e.g., representing spans for reading comprehension, semantic role labeling, or coreference resolution.
* `ArrayField` is an array representing some data that you have already converted into a matrix, e.g., images and hand-crafted feature vectors

`Fields` can be created simply by supplying the data. `Field` objects provide APIs for creating empty fields, counting vocabulary items, creating tensors, and batching tensors, among others. See the following code snippet for more detail.

<codeblock source="reading-textual-data/fields"></codeblock>

## Instances

An instance is the atomic unit of prediction in machine learning. In AllenNLP, `Instances` are collections of `Fields`, and datasets are collections of `Instances`.

`Instances` are created by dataset readers and used to create a `Vocabulary`. The `Vocabulary` is then used to map all strings in the `Instance`'s `Fields` into integer IDs, so that they can be turned into tensors. Later in the training pipeline, these tensors are batched together and fed to the model. The following diagram shows how `Fields` and `Instances` are created from a dataset.

<img src="/reading-textual-data/fields-and-instances.svg" alt="Fields and Instances" />

`Instances` can be created by passing a dictionary of field names and corresponding fields to the constructor. `Instances` know how to turn themselves into a dictionary of field names and corresponding tensors, which is then used by `Batches` to batch together tensors of the same type. See the following code snippet for how to create instances and use their APIs.

The fields names are important—because the resulting dictionary of tensors is passed by name to the model after being destructured, they have to match the model's `forward()` arguments exactly.

<codeblock source="reading-textual-data/instances"></codeblock>

</exercise>

<exercise id="2" title="Dataset readers">

## Basics of dataset readers

We already touched upon `DatasetReaders` and wrote our own in [Your first model](/your-first-model). `DatasetReaders` read datasets and convert them to collections of `Instances`. The `DatasetReader` class provides common interfaces to make it easier to write your own dataset reader, as well as extra features such as caching and lazy loading.

AllenNLP is shipped with a number of dataset reader implementations for common NLP datasets and tasks, including:

* `TextClassificationJsonReader` (text classification)
* `SequenceTaggingDatasetReader` (sequence labeling)
* `SimpleLanguageModelingDatasetReader` (language modeling)
* `SnliReader` (NLI)
* `SrlReader` (span detection)
* `Seq2SeqDatasetReader` (seq2seq)
* `PennTreeBankConstituencySpanDatasetReader` `UniversalDependenciesDatasetReader` (parsing)

You can implement your own dataset reader by subclassing the `DatsetReader` class. The code snippet below is the dataset reader we implemented in [Your first model](/your-first-model). The returned dataset is a list of `Instances` by default.

<codeblock source="reading-textual-data/dataset_reader_basic"></codeblock>

Dataset readers are designed to read data from a local file, although in some cases you may want to read data from an URL. AllenNLP provides an utility method `cached_path` to support this. If an URL is passed to the method it will download the resource to a local file and return its path. If you want your dataset reader to support both local paths and URLs, you can wrap `file_path` using `cached_path` in your `_read()` method as follows:

```python
from allennlp.common.file_utils import cached_path

...

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        with open(file_path, 'r') as lines:
            for line in lines:
```

## Lazy mode

Dataset readers also support reading data in a lazy manner, where a  `DatasetReader` yields instances as needed rather than returning a list of all instances at once. This comes in handy when your dataset is too big to fit into memory or you want to start training your model immediately.

When `lazy=True` is passed to a dataset reader's constructor, its `read()` method returns a `LazyInstances` object (instead of a list of `Instances`), which is a wrapper and an iterator that calls `_read()` and produces instances when called.

<codeblock source="reading-textual-data/dataset_reader_lazy"></codeblock>

# Caching dataset

<codeblock source="reading-textual-data/dataset_reader_cache"></codeblock>

Instances are serialized by `jsonpickle` by default, although you can override this behavior if you want.

</exercise>

<exercise id="3" title="Vocabulary">

* What is Vocabulary, how it fits with the rest of the pipeline
* (A diagram showing how Vocabulary works)
* What are Namespaces
* Getting indices from tokens (and vice versa)
* A word on min_count
* A word on non_padded_namespaces (the next exercise gives more details on padding)

</exercise>

<exercise id="4" title="Iterators, batching, and padding">

* How iterating and batching works in AllenNLP
* How padding works in AllenNLP
* (A diagram showing how batching works)
* Some details of bucket iterator (sorting_keys, maximum_samples_per_batch, etc.)

</exercise>
