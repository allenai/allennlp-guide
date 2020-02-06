---
title: 'Building your model'
description: 'This chapter gives a deep dive into one of the most important components of AllenNLP—Model—and an in-depth guide to building your own model.'
type: chapter
---

<exercise id="1" title="Model and its methods">

* Why we have more than just a PyTorch Module
* forward() and its return value
* Other Model methods (forward_on_instance(s), decode, get_metrics, etc.)

</exercise>

<exercise id="2" title="Abstractions for building models">

* Linear and FeedForward (?)
* Seq2SeqEncoder
* Seq2VecEncoder

</exercise>

<exercise id="3" title="Handling padding and masking">

* get_text_field_mask()
* sequence_cross_entropy_with_logits()

</exercise>
