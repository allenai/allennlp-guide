---
title: 'Reading textual data'
description:
prev: /next-steps
next: /building-your-model
type: chapter
id: 301
---

<exercise id="1" title="Fields and instances">

* Fields
    * what they are (= things that get converted to tensors), commonly used fields and how to use them
    * we have a separate chapter that gives a deep dive into TextFields and TextEmbedders
* Instances
    * what they are, how to create them, how they fit with the rest of the AllenNLP pipeline
    * (a diagram showing how fields and instances are related, with the rest of the pipeline)

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
