---
title: 'Chapter 3: Representing text as features: Tokenizers, TextFields, and TextFieldEmbedders'
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

In this exercise you can get some hands-on experience with how `TextField` data processing works.
We'll tokenize text and convert it into arrays using a `TextField`, printing out what it looks like
as we go.  First just run the code block below as is and see what comes out, then try modifying it
however you like to see what happens.

After you've gotten a feel for what's going on in the given example, as an exercise, see if you can
switch from using a single integer id to represent each word to representing words by the sequence
of characters that make them up.

<codeblock id="chapter03/data/simple">
You'll want to use a `TokenCharactersIndexer` instead of a `SingleIdTokenIndexer`, and you'll need
to update the vocabulary, also.
</codeblock>

This time, let's modify the same code, but to use characters as tokens.  Both of these cases use
characters as their base input, but if we do it this way, in the model we'll get a single vector
per _character_, instead of per _word_.

<codeblock id="chapter03/data/simple2">
You'll want to modify the `Tokenizer` and the `Vocabulary`.
</codeblock>

</exercise>

<exercise id="4" title="Combining multiple TokenIndexers">

What happens when you provide more than one TokenIndexer, how to make it work.

<codeblock id="chapter03/data/combined">
You'll want to modify the `Tokenizer` and the `Vocabulary`.
</codeblock>

</exercise>

<exercise id="5" title="Contextualized representations in TextFields">

ELMo and BERT, how they get converted to tensors, and how wordpieces are handled.

</exercise>

<exercise id="6" title="The model side: TextFieldEmbedders" type="slides">

<slides source="chapter03/06_model_side">
</slides>

</exercise>

<exercise id="7" title="Embedding simple TextField inputs">

Cover SingleIdTokenIndexer and TokenCharactersIndexer, point to other simple ones.

</exercise>

<exercise id="8" title="Embedding text that has multiple TokenIndexers">

What happens when you provide more than one TokenIndexer, how to make it work.

</exercise>

<exercise id="9" title="Embedding contextualized inputs">

ELMo and BERT, how they get converted to tensors, and how wordpieces are handled.

</exercise>
