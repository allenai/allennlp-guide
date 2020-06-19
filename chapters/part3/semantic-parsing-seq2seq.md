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


<exercise id="2" title="A toy task: Natural Language Arithmetic">

Let us now define a toy semantic parsing task. Taking inspiration from
[Liang and Potts, 2015](https://www.annualreviews.org/doi/pdf/10.1146/annurev-linguist-030514-125312)
and
[Bill MacCartney's example](https://nbviewer.jupyter.org/github/wcmac/sippycup/blob/master/sippycup-unit-1.ipynb)
used for a toy semantic parser, we'll look at the task of understanding
_Natural Language Arithmetic_. That is we will build a system that can translate natural language expressions like

```
seven times three minus two
eight over four
```

to arithmetic expressions like
```
7 * 3 - 2
8 / 4
```

A big advantage of working with a toy problem like this is that we do not have to deal with several important
challenges that exist in real-world semantic parsing problems.
First, the set of symbols here is finite and small. We only have
ten basic elements (or _entities_): ``zero``, ``one``, ``two``, ..., ``nine``, and four operators:
``plus``, ``minus``, ``times``, and ``over``. In real world problems like the ones we discussed at the beginning
of this chapter, you will have to deal with a potentially infinite set of entities and operators. For example,
you will want Alexa to be able to control any voice-enabled device, or be able to build natural language
interfaces for any table in your database.
Second, the meaning of your utterances are completely unambiguous in the case of the toy problem. For example, ``seven`` always means ``7``, whereas, "Turn on the lights" can refer to any set of lights depending on your location when you issue that command.


When we have expressions with multiple
operators in them, we have to decide the order in which we should perform those operations, since the result of the
computation depends on the order. For example, the result of ``7 * 3 - 2`` can either be ``19`` or ``7``
depending on whether we perform the subtraction or the multiplication first. To avoid this ambiguity, we will
use a bracketed [prefix notation](https://en.wikipedia.org/wiki/Polish_notation), making our targets look
like this:

```
(- ( * 7 3) 2)
(/ 8  4)
```

More formally, we will decide on a prespecified [order of operations](https://en.wikipedia.org/wiki/Order_of_operations)
to decide the nesting in the targets, and the system we will build for the task will have to learn the precedence
of operators, i.e., that `seven times three minus two` should be translated to `(- (* 7 3) 2)`, and not `(* 7 (- 3 2))`.

This [script](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/scripts/generate_data.py)
respects the conventional order of operations and produces natural language arithmetic statements paired with
corresponding arithmetic expressions, with an arbitrary number of operators per expression. We use it to generate
[train](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/data/nla_with_meaning_rep_train.tsv),
[validation](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/data/nla_with_meaning_rep_dev.tsv),
and [test](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/data/nla_with_meaning_rep_test.tsv) splits
with 1 to 10 operators in each expression.

## Defining metrics for the task

Given that we have defined the task, we are ready to specify how we will measure the progress of our models, i.e., the metrics
for the task. We covered defining metrics in [Building your model section](/building-your-model#2) of this guide.

For Natural Language Arithmetic, we can measure the quality of predictions using two metrics:

1. Well-formedness: Measures whether a given prediction has a sensisble order of operators and their arguments, and balanced parentheses. For example, ``( + 2 3 )`` is well-formed, and ``( + 3)`` and ``(+ 2 3`` are not. Note that this metric does not depend on the target.
2. Sequence accuracy: Measures whether the prediction and target are exactly the same. This is stricter than well-formedness because an accurate sequence is automatically well-formed.

We could have measured just the sequence accuracy alone, but it is useful to know whether the model is producing well-formed outputs
to check whether it has learned the ordering of operators, numbers and parentheses.

The following code shows an implementation of the metrics, and shows the values of the metrics for example inputs. Feel free to try your own.

<codeblock source="part3/semantic-parsing-seq2seq/metric_source" setup="part3/semantic-parsing-seq2seq/metric_setup"></codeblock>

</exercise>


<exercise id="3" title="Semantic parsing as machine translation">

The simplest way to look at the semantic parsing problem is as a
[typical](https://www.aclweb.org/anthology/N06-1056/)
[machine](https://www.aclweb.org/anthology/P13-2009/)
[translation](https://www.aclweb.org/anthology/P16-1002/)
[problem](https://www.aclweb.org/anthology/P16-1004/).  Instead of translating from one natural
language to another, we will translate from a natural language to a programming language.  In 2020,
that means using a seq2seq model to generate a program conditioned on some input utterance.

We don't have a chapter on seq2seq models yet, but
[here](https://nlp.stanford.edu/~johnhew/public/14-seq2seq.pdf) is a good overview of the concepts
involved.  We will encode the input utterance using some encoder, then decode a sequence of tokens
in the target (programming) language.


</exercise>


<exercise id="4" title="Implementing a seq2seq model">

For this, we are literally just taking AllenNLP's existing seq2seq model and using it as-is for
semantic parsing.  We'll highlight a few relevant points here, but we will defer most details to the
chapter on general seq2seq models (which isn't currently written).

## DatasetReader

The code example below shows a simplified version of a `DatasetReader` for seq2seq data.  We just
have two `TextFields` in our `Instance`, one for the input tokens and one for the output tokens.
There are two important things to notice:

1. We are using different vocabulary namespaces for our input tokens and our output tokens.  The
   reason we are doing this is becaus this is the easiest way to get a vocabulary for all allowable
output tokens in the programming language, which we can use in the model's decoder.  This also has
the side effect of giving a separate embedding for any overlapping tokens in the input and output
language.  An open parenthesis, or the number 2, will get different embeddings if it's in the input
or the output.  Depending on your task, this can be a good thing or a bad thing.  There are ways to
get around this side effect and share your embeddings, but they are more complicated.  Feel free to
ask a question in our [Discourse forum](https://discourse.allennlp.org) or open an issue on the
[guide repo](https://github.com/allenai/allennlp-guide) if you would like to see some more detail on
how to do this.
2. We're adding special tokens to our output sequence.  In a seq2seq model, you typically give a
   special input to the model to tell it to start decoding, and it needs to have a way of signaling
that it is finished decoding.  So, we modify the program tokens to include these signaling tokens.

<codeblock source="part3/semantic-parsing-seq2seq/dataset_reader_source" setup="part3/semantic-parsing-seq2seq/dataset_reader_setup"></codeblock>

It is not shown here, but in AllenNLP's standard training loop, the source and target vocabularies
will be created by iterating once over the data.  If you have special vocabulary considerations that
aren't handled nicely in this way, see the [`Vocabulary` section](/reading-data#3) of this guide.

## Model


</exercise>


<exercise id="5" title="Training">

</exercise>


<exercise id="6" title="Decoding">

</exercise>


<exercise id="7" title="Further reading">

In [section 3](#3) we gave links to a series of papers that used standard translation techniques at
the time to approach semantic parsing problems.  There are a lot of variations on standard seq2seq
approaches for semantic parsing, including [recursive or two-stage
generation](https://www.aclweb.org/anthology/P18-1068/) and [grammar-based
decoding](/semantic-parsing-grammar).  AllenNLP has strong support for grammar-based decoding, and
this is the subject of the next chapter.

</exercise>
