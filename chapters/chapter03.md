---
title: 'Chapter 3: TextFields'
description:
  "A deep dive into AllenNLP's core abstraction: how exactly we represent textual inputs, both on
the data side and the model side."
prev: /chapter02
next: /chapter04
type: chapter
id: 3
---

<textblock>

This chapter assumes you have a basic familiarity with what a `Field` is in AllenNLP, which you can
get near the beginning of [chapter 2 of this course](/chapter02).

</textblock>


<exercise id="1" title="The basic problem: text to features" type="slides">

<slides source="chapter03/01_the_basic_problem">
</slides>

</exercise>

<exercise id="2" title="The data side: TextFields" type="slides">

<slides source="chapter03/02_data_side">
</slides>

</exercise>

<exercise id="3" title="A simple TextField example">

Cover SingleIdTokenIndexer and TokenCharactersIndexer, point to other simple ones.

</exercise>

<exercise id="4" title="Contextualized representations in TextFields">

ELMo and BERT, how they get converted to tensors, and how wordpieces are handled.

</exercise>

<exercise id="5" title="Combining multiple TokenIndexers">

What happens when you provide more than one TokenIndexer, how to make it work.

</exercise>

<exercise id="6" title="The model side: TextFieldEmbedders" type="slides">

<slides source="chapter03/06_model_side">
</slides>

</exercise>

<exercise id="7" title="Embedding simple TextField inputs">

Cover SingleIdTokenIndexer and TokenCharactersIndexer, point to other simple ones.

</exercise>

<exercise id="8" title="Embedding contextualized inputs">

ELMo and BERT, how they get converted to tensors, and how wordpieces are handled.

</exercise>

<exercise id="9" title="Embedding text that has multiple TokenIndexers">

What happens when you provide more than one TokenIndexer, how to make it work.

</exercise>
