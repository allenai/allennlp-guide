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

A `Field` contains a piece of data of particular type. `Fields` get converted to a tensor in a model, either as an input or an output, after being converted to IDs and batched & padded. 

Note that an AllenNLP `Field` represents a piece of data for one `Instance` (see below for more about `Instances`). This is different from e.g., [torchtext](https://torchtext.readthedocs.io/), where a `Field` represents all pieces of data of a particular type for a dataset.

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

`Instances` are created by dataset readers and used to create a `Vocabulary`. Later in the training pipeline, `Instances` are batched together, turned into tensors, and fed to the model. The following diagram shows how `Fields` and `Instances` are created from a dataset.

<img src="/reading-textual-data/fields-and-instances.svg" alt="Fields and Instances" />

`Instances` can be created by passing a dictionary of field names and corresponding fields to the constructor. `Instances` know how to turn themselves into a dictionary of field names and corresponding tensors, which is then used by `Batches` to batch together tensors of the same type. See the following code snippet for how to create instances and use their APIs.

<codeblock source="reading-textual-data/instances"></codeblock>

</exercise>

<exercise id="2" title="Dataset readers">

* What are dataset readers (recap of Part 1)
* Common dataset readers (mainly pointers to Part 3)
* Lazy mode
* A word on instance/dataset caching
* A word on common.file_utils.cached_path

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
