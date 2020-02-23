---
title: 'Common architectures'
description: "In this chapter we'll introduce neural architectures and AllenNLP abstractions that are commonly used for building your NLP model."
type: chapter
---

<textblock>
The main modeling operations done on natural language inputs include summarizing sequences, contextualizing sequences (that is, computing contextualized embeddings from sequences), modeling spans within a longer sequence, and modeling similarities between sequences using attention. In the following sections we'll learn AllenNLP abstractions for these operations.
</textblock>

<exercise id="1" title="Summarizing sequences">

* Seq2VecEncoder
    * RNN, CNN
* Sample code

</exercise>

<exercise id="2" title="Contextualizing sequences">

* Seq2SeqEncoder
    * RNN
* Sample code

</exercise>

<exercise id="3" title="Modeling spans in sequences">

* SpanField
* SpanExtractor
* Sample code
* pruning

</exercise>

<exercise id="4" title="Modeling similarities between sequences">

* Attention
* MatrixAttention
    * Why two abstractions for attention
* Sample code
    * Similarity matrix computation in BiDAF

</exercise>

<exercise id="5" title="Common neural network techniques">

* FeedForward
* Activations
    * Sample code
* ConditionalRandomField
* Highway and residual connection
* TimeDistributed
* GatedSum

</exercise>
