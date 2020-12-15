---
title: 'Semantic Parsing: Grammar-based Decoding'
description:
  "Here we build on the prior semantic parser chapter and build a better model, using a technique called grammar-based decoding."
type: chapter
---

<exercise id="1" title="Semantic parsing recap">

In the previous chapter we introduced semantic parsing as the task of translating natural language utterances into executable programs.
Recall that we focused on the toy task of Natural Language Arithmetic, that involved translating sequences like
```
three times four plus seven minus five
```
into
```
(subtract (add (multiply 3 4) 7) 5)
```

We took an initial step towards building a semantic parser by viewing the problem as a sequence-to-sequence translation task: the model
encoded the input utterance as a sequence of tokens in natural language (`three`, `times`, ...), and decoded the program as a sequence of elements
in the domain specific programming language (`(`, `subtract`, `(`, ...).

Based on this view, we trained a `Seq2seq` model for the task. Towards the end of the chapter we noted that the model is prone to producing
illegal outputs, particularly when the inputs are long with more than five operators.

We suggested that this issue can be fixed by placing constraints on the output based on the rules of the target programming language. This idea
is based on a key insight. We are using modeling techniques inspired by machine translation, but there is an important difference between machine
translation and semantic parsing: since the target languages in semantic parsing are not natural, they obey certain prespecified rules without
exceptions. The idea here is to leverage those rules to disallow invalid outputs. Note that the same cannot be done in the case of generating
natural languages, because no matter how complex you make the rules, there are bound to be exceptions due to the nature of those languages.

</exercise>


<exercise id="2" title="Grammar-based decoding">

```
S -> val
val -> (op val val)
op -> add | subtract | multiply | divide
val -> 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
```

</exercise>


<exercise id="3" title="Defining a domain-specific (target) language">

</exercise>


<exercise id="4" title="Transition functions">

</exercise>


<exercise id="5" title="State tracking">

</exercise>


<exercise id="6" title="Training">

</exercise>


<exercise id="7" title="Decoding">

</exercise>


<exercise id="8" title="Further reading">

</exercise>
