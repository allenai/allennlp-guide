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
ten basic elements (or _entities_): `zero`, `one`, `two`, ..., `nine`, and four operators:
`plus`, `minus`, `times`, and `over`. In real world problems like the ones we discussed at the beginning
of this chapter, you will have to deal with a potentially infinite set of entities and operators. For example,
you will want Alexa to be able to control any voice-enabled device, or be able to build natural language
interfaces for any table in your database.
Second, the meaning of your utterances are completely unambiguous in the case of the toy problem. For example, `seven` always means `7`, whereas, "Turn on the lights" can refer to any set of lights depending on your location when you issue that command.


When we have expressions with multiple
operators in them, we have to decide the order in which we should perform those operations, since the result of the
computation depends on the order. For example, the result of `7 * 3 - 2` can either be `19` or `7`
depending on whether we perform the subtraction or the multiplication first. To avoid this ambiguity, we will
use a bracketed [prefix notation](https://en.wikipedia.org/wiki/Polish_notation), making our targets look
like this:

```
(- (* 7 3) 2)
(/ 8  4)
```

More formally, we will decide on a prespecified [order of operations](https://en.wikipedia.org/wiki/Order_of_operations)
to decide the nesting in the targets, and the system we will build for the task will have to learn the precedence
of operators, i.e., that `seven times three minus two` should be translated to `(- (* 7 3) 2)`, and not `(* 7 (- 3 2))`.

This [script](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/scripts/generate_data.py)
respects the conventional order of operations and produces natural language arithmetic statements paired with
corresponding arithmetic expressions, with an arbitrary number of operators per expression. Instead of using arithmetic symbols
like `+`, `-`, the script uses operator names like `add` and `subtract` to be able to define them as Python functions. Making
these Python functions is desirable since we'll use the typing information to instantiate a grammar from these functions later.
We'll cover those details later, but all you need to know is that the targets look like the following:

```
(subtract (multiply 7 3) 2)
(divide 8 4)
```

We use the script to generate
[train](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/data/nla_with_meaning_rep_train.tsv),
[validation](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/data/nla_with_meaning_rep_dev.tsv),
and [test](https://github.com/allenai/allennlp-guide-examples/blob/master/nla_semparse/data/nla_with_meaning_rep_test.tsv) splits
with 1 to 10 operators in each expression.

## Defining metrics for the task

Given that we have defined the task, we are ready to specify how we will measure the progress of our models, i.e., the metrics
for the task. We covered defining metrics in [Building your model section](/building-your-model#2) of this guide.

For Natural Language Arithmetic, we can measure the quality of predictions using two metrics:

1. Well-formedness: Measures whether a given prediction has a sensisble order of operators and their arguments, and balanced parentheses.
For example, `(add 2 3)` is well-formed, and `(add 3)` and `(add 2 3 ` are not. Note that this metric does not depend on the target.
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

## Dataset Reader

The code example below shows a simplified version of a `DatasetReader` for seq2seq data.  We just
have two `TextFields` in our `Instance`, one for the input tokens and one for the output tokens.
There are two important things to notice:

1. We are using different vocabulary namespaces for our input tokens and our output tokens.  The
   reason we are doing this is because this is the easiest way to get a vocabulary for all allowable
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

Let us now look at the key parts of the `Seq2seq` model. A Seq2seq model typically involves two sequential modules,
like recurrent neural networks with LSTM cells, one that processes the input sequence one token at a time, and another
that outputs a predicted output sequence, one token at a time. These two modules are usually called encoder and decoder respectively.
The following is a high-level gist of how a Seq2seq model is trained (with two caveats):

1. We embed the source tokens and encode the sequence using a sequence encoder.
2. We initialize a sequence decoder with the final state of the encoder and use the decoder to run the following steps as many
times as we want the maximum length of the output sequence to be:
    - 2a. Embed the token output by the decoder from the previous step.
    - 2b. Decode (or more precisely, encode with the decoder) the embedded output.
    - 2c. Classify the decoded output to predict an index into the target vocabulary. The predicted index indicates the output
that will be embedded in the next decoding step.
3. We compute a loss by comparing the predicted sequence of tokens against the target tokens.

Now for the caveats. In practice, to make this process work, two modifications are typically made.

### Attention

In the process described above, the only information the decoder gets about the input sequence is in the form of the final state
of the encoder after it processes the input sequence. Decoding the entire output sequence from just this information can be very difficult.
To make it easier, Seq2seq models use a so called [attention](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
mechanism that lets the decoder access a summary of the outputs of the encoder after processing each input token. The summary itself is computed
based on the current state of the decoder. (TODO: pointers to code)

If using attention, we instead decode a concatenation of the embedded output, and a summary of the enoded source sequence.

### Scheduled Sampling



</exercise>


<exercise id="5" title="Training">

We're now ready to actually train the model. We need to define a configuration to specify the
attributes of the model, dataset reader, and the trainer and locations of the training and
validation datasets. See the [chapter on configuration files](/using-config-files) of this guide
for more details. This is the configuration we'll use:

```
{
  "dataset_reader": {
    "type": "seq2seq",
    "source_tokenizer": {
      "type": "whitespace"
    },
    "target_tokenizer": {
      "type": "whitespace"
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "namespace": "target_tokens"
      }
    }
  },
  "train_data_path": "data/nla_with_meaning_rep_train.tsv",
  "validation_data_path": "data/nla_with_meaning_rep_dev.tsv",
  "model": {
    "type": "composed_seq2seq",
    "source_text_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "embedding_dim": 100,
          "trainable": true
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 100,
      "hidden_size": 50,
      "num_layers": 1
    },
    "decoder": {
      "decoder_net": {
         "type": "lstm_cell",
         "decoding_dim": 50,
         "target_embedding_dim": 50,
         "attention": {
           "type": "dot_product"
         }
      },
      "max_decoding_steps": 50,
      "target_namespace": "target_tokens",
      "target_embedder": {
        "vocab_namespace": "target_tokens",
        "embedding_dim": 50
      },
      "scheduled_sampling_ratio": 0.5,
      "beam_size": 10,
      "token_based_metric": "nla_metric"
    }
  },
  "data_loader": {
    "batch_sampler": {
        "type": "bucket",
        "batch_size": 10,
        "padding_noise": 0.0
    }
},
  "trainer": {
    "num_epochs": 20,
    "patience": 10,
    "validation_metric": "+sequence_accuracy",
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}
```

You can train the model on your machine by cloning [the examples repository](https://github.com/allenai/allennlp-guide-examples), and
going to the `nla_semparse` directory.

```
cd allennlp-guide-examples/nla_semparse
```

The following line installs `allennlp` and `allennlp_models`.
```
pip install -r requirements.txt
```

And the following line trains the model.
```
allennlp train training_config/seq2seq_config.json -s /tmp/nla_seq2seq \
--include-package allennlp_models --include-package nla_semparse
```

Training should take roughly one minute per epoch on a Macbook. You'll see in the configuration file that we're tracking the `sequence_accuracy`
metric, which we discussed earlier, on the validation set for early stopping.

At the end of training, you'll see the following numbers for the validation set for metrics from the best epoch.

```
"best_validation_well_formedness": 0.581
"best_validation_sequence_accuracy": 0.242
```

You can measure the performance of the model on the test set by running the following command

```
allennlp evaluate /tmp/nla_seq2seq/model.tar.gz data/nla_with_meaning_rep_test.tsv \
--include-package allennlp_models --include-package nla_semparse
```

and it should give you the following numbers

```
"well_formedness": 0.602
"sequence_accuracy": 0.245
```

</exercise>


<exercise id="6" title="Decoding">

Let us now look at what kinds of predictions the model makes.
We'll use a [`Predictor`](/training-and-prediction#4) to do so, and make predictions on various kinds of inputs.
For our current purpose, we can directly use the
[`Seq2SeqPredictor`](http://docs.allennlp.org/models/master/models/generation/predictors/seq2seq/) implemented in
the `allennlp-models` repository, with the model archive we trained on our task.

What that means is we'll load the weights that resulted from the training process, and the task-specific vocabulary from the archive, and instantiate
a `Seq2Seq` predictor with those weights and vocabulary. The predictor uses the dataset reader we defined for the task, and passes each new NLA
expression we want to decode through the `text_to_instance` method of our dataset, the output of which will be passed through the `forward` method in
our model instantiated with the trained weights. The output will be the model's top prediction, i.e., the sequence with the maximum `p(y | x; $\theta$)`, where `y` is an artithmetic expression, `x` is the input NLA expression, and `$\theta$` are the weights of the `Seq2Seq` model. Of course, all of this
is hidden behind a single `predictor.predict_json()` call below.

<codeblock source="part3/semantic-parsing-seq2seq/predictor_source_easy"></codeblock>

Looks like the model's doing pretty well on these inputs. It has clearly learned the mapping from strings to symbols, and the order of operators and their arguments.

Let's try something more interesting now.

<codeblock source="part3/semantic-parsing-seq2seq/predictor_source_medium" setup="part3/semantic-parsing-seq2seq/predictor_setup"></codeblock>

This is beginning to look impressive. Not only has the model learned basic nesting correctly, it has also learned something about the precedence of operators that was followed in the training data.

Let's push the model harder.

<codeblock source="part3/semantic-parsing-seq2seq/predictor_source_hard" setup="part3/semantic-parsing-seq2seq/predictor_setup"></codeblock>

Now we see some problems. Both the predicted expressions are ill-formed. These may be a little hard to read, but the first prediction is essentially
`(subtract (multiply ..) (add ..)) 1)`, with a lone `1)` at the end of a complete expression, and the second from last `add` in the second prediction
does not have a second argument.

Recall that the `well_formedness` measure on the test set for this model was `0.602`, meaning that about `40%` of the predictions made by
this model on expressions with 1 to 10 operators (which was how generated this dataset) are expected to be ill-formed. It looks like the model struggles
particularly with expressions having more operators.

How can we fix this issue? One way is to increase the capacity of the model (by adding more parameters), and/or train it longer. But that still
does not guarantee that the model produces well-formed outputs with a much larger number of operators. Let's consider another option.

Note that at every step while producing the output sequence, the `Seq2Seq` model chooses between tokens like `(`, `add`, `7` etc. Many of these options
are illegal according our rules of Natural Language Arithmetic. For example, after producing a `(`, the model does not need to even consider the
option of producing a number, since any valid expression will only have an operator in that position. We will explore this option in the next chapter.

</exercise>


<exercise id="7" title="Further reading">

In [section 3](#3) we gave links to a series of papers that used standard translation techniques at
the time to approach semantic parsing problems.  There are a lot of variations on standard seq2seq
approaches for semantic parsing, including [recursive or two-stage
generation](https://www.aclweb.org/anthology/P18-1068/) and [grammar-based
decoding](/semantic-parsing-grammar).  AllenNLP has strong support for grammar-based decoding, and
this is the subject of the next chapter.

</exercise>
