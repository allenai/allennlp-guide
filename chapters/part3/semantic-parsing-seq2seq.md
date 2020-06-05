---
title: 'Semantic Parsing: Intro and Seq2Seq Model'
description:
  "This chapter describes the problem of semantic parsing—mapping language to executable programs—and how to build a simple seq2seq semantic parser with AllenNLP."
type: chapter
---

<exercise id="1" title="Semantic parsing">

If you have ever interacted with a voice assistant like Siri or Alexa, then you have used a semantic
parser.  _Semantic parsing_ is our term for translating natural language statements into some
executable meaning representation.[^1]

[^1]: "Semantic parsing" is also used to refer to _non-executable_ meaning representations, like AMR
  or semantic dependencies. In this series of chapters on semantic parsing, we're referring
  exclusively to the executable kind of meaning representation.

<img src="/part3/semantic-parsing/alexa-as-semantic-parser.svg" alt="Alexa as a semantic parser" />

Semantic parsers form the backbone of voice assistants, as shown above, or they can be used to
answer questions or give natural language interfaces to databases,

<img src="/part3/semantic-parsing/database-interface.svg" alt="Natural language interface to databases as semantic parsers" />

or even write general-purpose code:

<img src="/part3/semantic-parsing/outputting-code.svg" alt="Outputting python code with semantic parsing" />

In this last case, there isn't an "execution" of the python class without more context, but, if we
did a good job parsing the question into code, the code that we output should in fact be executable
in the right context.

This chapter, and a series of chapters to follow, will talk about how to implement a parser that
converts natural language statements to some executable piece of code.

### Representing this as an NLP problem

To actually learn a semantic parsing model, we first need to make some assumptions about what kind
of data we're working with.  We'll start by assuming that we have a collection of input utterances
paired with labeled programs, which we call the _fully supervised_ setting, because our parser is
trying to output programs and can learn directly from the labeled programs that we give it at
training time.  Our model will get as input a natural language utterance, and will be expected to
produce as output a program in some programming language of choice.

We also need to know something about the programs that we're outputting.  A key feature of the
semantic parsing problem that the best models make use of is that there are particular constraints
on the output—we know that the output should be a valid program in some programming language, and we
can write down a set of hard constraints on what is a valid program.  We'll start with a very simple
model in this chapter which doesn't take into account these constraints, but the rest of the
chapters will leverage the output programming language to build stronger models.

</exercise>


<exercise id="2" title="Some toy data: natural language arithmetic">

</exercise>


<exercise id="3" title="Semantic parsing as machine translation">

</exercise>


<exercise id="4" title="Implementing a seq2seq model">

</exercise>


<exercise id="5" title="Training">

</exercise>


<exercise id="6" title="Decoding">

</exercise>


<exercise id="7" title="Further reading">

Mirella Lapata and students, Jia and Liang, ...

</exercise>
