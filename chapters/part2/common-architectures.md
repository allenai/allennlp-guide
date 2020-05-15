---
title: 'Common architectures'
description: "In this chapter we'll introduce neural architectures and AllenNLP abstractions that are commonly used for building your NLP model."
type: chapter
---

<textblock>

The main modeling operations done on natural language inputs include summarizing sequences,
contextualizing sequences (that is, computing contextualized embeddings from sequences), modeling
spans within a longer sequence, and modeling similarities between sequences using attention. In the
following sections we'll learn AllenNLP abstractions for these operations.

</textblock>


<exercise id="1" title="Summarizing sequences">

Taking a sequence of tokens and summarizing it to a fixed-size vector is one of the most fundamental
operations done on natural language inputs. AllenNLP provides an abstraction called `Seq2VecEncoder`
for this, which is a class of architectures that take a sequence of vectors and summarize it to a
single vector of fixed size. It abstracts any operation that takes a tensor of shape `(batch_size,
sequence_length, input_size)` and produces another of shape `(batch_size, output_size)`. This
includes a wide range of models, from something very simple (a bag-of-embedding model which simply
sums up the input embeddings) to something more complicated (a pooling layer of BERT). See the
following diagram for an illustration:

<img src="/part2/common-architectures/seq2vec.svg" alt="Seq2Vec encoder" />

RNNs are a popular choice for summarizing sequences in many NLP models. Instead of implementing its
own RNN-based `Seq2VecEncoders`, AllenNLP offers `PytorchSeq2VecWrapper`, which wraps PyTorch's
existing RNN implementations (such as `torch.nn.LSTM` and `torch.nn.GRU`) and make them compatible
with AllenNLP's `Seq2VecEncoder` interface. In most cases, though, you don't need to use the wrapper
yourself—wrapped `Seq2VecEncoders` are already defined by AllenNLP as e.g., `LstmSeq2VecEncoder` and
`GruSeq2VecEncoder`.

Other commonly used `Seq2VecEncoders` include `CnnEncoder`, which is a combinations of convolutional
layers and a pooling layer on top of them, as well as `BertPooler`, which returns the embedding for
the `[CLS]` token of the BERT model.

We already used a `Seq2VecEncoder` when we built [a sentiment classifier in Part
1](your-first-model#3). Remember that we defined the constructor of `SimpleClassifier` to take any
`Seq2VecEncoder`:

<pre data-line="6,9" class="language-python line-numbers"><code>
@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
</code></pre>

This allows you to plug in any `Seq2VecEncoder` implementations when defining the model. For
example, you can do:

```python
...
encoder = LstmSeq2VecEncoder(input_size=5, hidden_size=2, num_layers=1)
model = SimpleClassifier(vocab, embedder, encoder)
...
```

Your model will use a 1-layer LSTM with `input_size=5` and `hidden_size=2` for summarizing the input
sequence.

In the following example code, we instantiate two different `Seq2VecEncoders` (LSTM and CNN) and
feed them a tensor of shape `(batch_size, sequence_length, input_size)` as input. Notice that you
get an output tensor of shape `(batch_size, output_size)` no matter what `Seq2VecEncoder` you use.

<codeblock source="part2/common-architectures/seq2vec"></codeblock>

Note that these `Seq2VecEncoders` operate on batched, padded input. A single batch may contain
sequences of different lengths, and shorter sequences get padded so that the batch has a uniform
shape. RNN-based `Seq2VecEncoders` need to know the length of each sequence in the batch in order to
return the correct hidden states. In AllenNLP, we use use masks to pass this information to the RNN.
A mask is simply a tensor of 0s and 1s that indicate which locations of the batch are padded and
non-padded. We'll discuss padding and masking more in details in [Representing text as
features](/representing-text-as-features).

</exercise>


<exercise id="2" title="Contextualizing sequences">

In the previous section, we covered `Seq2VecEncoders`, which abstract an operation for summarizing
sequences, but almost as common is a situation where you want to contextualize sequences, that is,
process a sequence of tokens and obtain another sequence of some embeddings. AllenNLP provides
`Seq2SeqEncoder`, which abstracts any operation that takes a tensor of shape `(batch_size,
sequence_length, input_dim)` and produces another, modified tensor of shape `(batch_size,
sequence_length, output_dim)`. This can be something as simple as returning the input tensor
unchanged (which is what the `PassThroughEncoder` does) or something more complicated such as the
Transformer Encoder.

<img src="/part2/common-architectures/seq2seq.svg" alt="Seq2Seq encoder" />

As with `Seq2VecEncoders`, AllenNLP provides a convenient class `PytorchSeq2SeqWrapper` that wraps
PyTorch-based RNNs and turns them into AllenNLP-compatible `Seq2SeqEncoders`. But again, you don't
need to use the wrapper yourself—AllenNLP provides pre-defined `Seq2SeqEncoders` such as
`LstmSeq2SeqEncoder` and `GruSeq2SeqEncoder`.

You might want to stack multiple `Seq2SeqEncoders` on top of each other and apply them in sequence.
For example, you might want to contextualize the input using an `LstmSeq2SeqEncoder` first then
further transform it using a `FeedForwardEncoder`, which applies `FeedForward` to each item in the
sequence. AllenNLP offers a seq2seq encoder called `ComposeEncoder` which does exactly this—it takes
a list of `Seq2SeqEncoders` and applies them in sequence.

In the following code example, we instantiate two different `Seq2SeqEncoders` and observe the shapes
of the input and the output tensors. The first two dimensions are unchanged (`batch_size` and
`sequence_length`) but the size of the output embeddings depends on the specific module you are
using. Note again that `Seq2SeqEncoders` take `mask` as an argument to `forward()`. RNN-based
seq2seq encoders need to know the padded and un-padded locations in order to properly handle the
statefulness (that is, using the final state of a batch as the inital state of the next one).

<codeblock source="part2/common-architectures/seq2seq"></codeblock>

## Why not just use torch.nn.LSTM (or similar) directly?

If this does what you want, go for it, there's nothing wrong with using LSTM modules directly in
your model.  The reasons you might want to use a `Seq2SeqEncoder` instead are two-fold: first, it
encourages you to think at a higher level about what basic operations your model is doing (am I
contextualizing, summarizing, or both?).  Second, it allows you to do controlled experiments easier,
if you think you might one day want to try a different contextualizer in your model.  Using an
abstraction that encapsulates the options you want to experiment with is a powerful way to get very
easy, controlled experiments.

Note also that we've now seen two different abstractions for RNNs—RNN for summarizing
(`Seq2VecEncoder`) and RNN for contextualizing (`Seq2SeqEncoder`). Although they are implemented in
a very similar way (they both use PyTorch's RNN implementations), they are conceptually different,
since the class of possible replacements for the former (e.g., CNN) is different from the that for
the latter (e.g., Transformer encoder). This is one example of how AllenNLP designs
abstractions—they abstract *what* is done to *what*, instead of *how* it's done.

Some pre-trained contextualizers (including BERT) are implemented as `TokenEmbedders` instead of
`Seq2SeqEncoders`. We'll cover these [in the next chapter](representing-text-as-features).

</exercise>


<exercise id="3" title="Modeling spans in sequences">

Representing spans is also a very powerful (and underused) abstraction in modern NLP that has a wide
range of applications including constituency parsing, reading comprehension, and coreference
resolution. AllenNLP provides convenient abstractions for representing and embedding spans, along
with some utility methods that make it easier to work with spans.

## Representing spans

Spans are usually represented as pairs of integers, one for the start and the other for the end
indices. AllenNLP uses `SpanFields`, which are a type of `Fields`, to represent such pairs. In order
to instantiate a `SpanField`, you need to provide a start index, an end index, as well as a
`SequenceField` (such as a `TextField`) that these indices are referring to, which is used to
validate whether the span is well defined. The start/end indices are inclusive. `SpanFields` are
converted to a tensor of size `(batch_size, 2)` when batched.

In many cases, however, a single instance can have an arbitrary number of spans. For example, a
single parse tree can contain multiple spans over the text corresponding to multiple constituents.
You can use `ListFields`, a type of `Fields` that represent lists of other fields, to put spans
together into a single field. When batched, these fields are converted to a tensor of size
`(batch_size, num_spans, 2)`. You can use the utility `enumerate_spans()` method in
`allennlp.data.dataset_readers.dataset_utils.span_utils` to enumerate all spans up to some fixed
length. See the code example below for an illustration.

[`PennTreeBankConstituencySpanDatasetReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/penn_tree_bank.py#L40)
and
[`ConllCorefReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/coreference_resolution/conll.py#L19)
are two dataset readers that generate instances with spans.

## Embedding spans

The next step for modeling spans in sequences is to embed them. This is usually done first by
contextualizing the input sequence by using e.g., a `Seq2SeqEncoder`, then by applying some
operation on the embeddings at/between the start and end indices. AllenNLP abstracts this as
`SpanExtractors`, which take a tensor of shape `(batch_size, sequence_length, embedding_dim)` and a
span tensor of shape `(batch_size, num_spans, 2)`, and return a tensor of size `(batch_size,
num_spans, encoding_dim)` where `encoding_dim` depends on the specific operation applied to embed
the spans. The figure below illustrates how `SpanExtractors` work.

<img src="/part2/common-architectures/spans.svg" alt="Embedding spans" />

`EndpointSpanExtractors`, the simplest type of `SpanExtractors`, embed spans using a combination of
the embeddings at the end points, as its name suggests. They take the embeddings of the end points
and apply some element-wise operation (such as multiplication or subtraction) to them.
`SelfAttentiveSpanExtractors`, another type of `SpanExtractors`, compute span representations first
by generating unnormalized attention scores for input tokens, then by taking a weighted sum of the
embeddings inside the span with respected to the normalized scores.

In the code example below, we first create a toy instance that contains a list of `SpanFields`, then
compute a span representation using an `EndpointSpanExtractor`. We observe the shapes of the input
tensors as well as the output when using different operations.

<codeblock source="part2/common-architectures/span"></codeblock>

## Pruning spans

Many span-based models involve some kind of enumeration of spans and need to consider all possible
spans in a given sentence or document. The number of possible spans is quadratic (O(n^2)) to the
length of the input, so in practice it is important to keep only the promising spans by *pruning*
them.

There are many possible ways to prune spans. One (probably the easiest) way is to filter them using
heuristics in your `DatasetReader`. For example, for the co-reference task, mentions rarely span
across sentences. In many tasks, it is often practically sufficient to consider spans of length up
to some small number of tokens. For this, you can use the `enumerate_spans()` utility method
mentioned above, which lets you enumerate all valid spans by providing a maximum and minimum span
width, as well as a boolean function to determine if any given span should be included. See the
example in the code snippet for how to use this method.

However, you cannot prune all the unnecessary spans just by using heuristics in your `DatasetReader`
this way. A common practice is to score span candidates in your model and only use the top *k* spans
for loss computation, disregarding the rest. 

You can use a learned function (usually a feedforward network) to score span candidates. After
feeding the span embeddings to the scoring network and obtaining a score per span, you can use the
[`masked_topk()`](https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L1705) utility
method in the `allennlp.nn` module to extract only the top-k items.

</exercise>


<exercise id="4" title="Modeling similarities between sequences">

Many modern NLP models make use of attention to compute similarities (some sort of relatedness
score) between sequences. For example, Seq2Seq models, for which the attention mechanism was first
invented, use attention to model similarities between the encoder and the decoder hidden states.

AllenNLP provides two types of abstractions for attention—`Attention` and `MatrixAttention`. We'll
discuss these two in detail below.

## Attention

AllenNLP's `Attention` modules compute similarities between (each row of) a matrix of size
`(batch_size, sequence_length, embedding_dim)` and a vector of size `(batch_size, embedding_dim)`
and return a (typically normalized) similarity vector of size `(batch_size, sequence_length)`. For
Seq2Seq models, for example, the input matrix is a sequence of encoder hidden states, and the input
vector is the decoder hidden state. 

There are many ways to compute the similarity between two vectors, and in AllenNLP each similarity
method has its own subclass of `Attention`. One of the simplest implementations is
`DotProductAttention`, which simply computes the dot product between the  vector and each row of the
matrix. The embedding dimension of these two inputs must be the same size (because it's just a dot
product).

A more general implementation is `BilinearAttention`, which computes `x^T W y + b` for a given
vector `x` and a matrix `y` using a matrix of weights `W` and a bias `b`. The sizes of the embedding
dimensions do not need to match.

Another general type of operation is implemented by `LinearAttention`, which computes a dot product
between a weight vector and a configurable combination of the two input vectors, followed by an
optional activation function. For example, if you specify `x,y,x*y` as the combination, this will
compute `w^T [x; y; x*y] + b` where `[;]` is vector concatenation and `*` is elementwise
multiplication. You can replicate [Bahdanau et al.
2016](https://www.semanticscholar.org/paper/Neural-Machine-Translation-by-Jointly-Learning-to-Bahdanau-Cho/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5)'s
attention by specifying `combination='x,y'` and a tanh activation, for example.

## MatrixAttention

AllenNLP provides another type of attention abstraction called `MatrixAttention`. `MatrixAttention`
takes two matrices of size `(batch_size, sequence_length1, embedding_dim)` and `(batch_size,
sequence_length2, embedding_dim)` and returns a matrix of size `(batch_size, sequence_length1,
sequence_length2)`, which contains the similarity between each row of two input matrices. As with
`Attention`, the specific type of operation applied to the inputs depends on the  implementation.

You may be wondering why there are two different abstractions for attention, even though they do
something very similar? The main reason is efficiency—there are many tasks where you may want to
model similarities between two sequences of vectors directly, and using tensor operations specific
to that case is more efficient. One such example is the [BiDAF (Bidirectional Attention Flow)
model](https://github.com/allenai/allennlp-models/blob/master/allennlp_models/rc/bidaf/bidaf_model.py)
for reading comprehension, where similarities between a passage and a question are modeled using a
`MatrixAttention` module:

```
# Shape: (batch_size, passage_length, question_length)
passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
```

Note that there are some subtle differences between `Attention` and `MatrixAttention`. The latter
does not normalize the output and thus does not take a mask. This is because there are a few
different ways you could decide to normalize the output; it's up to the caller to do any desired
(masked) normalization.

In the following code snippet, we instantiate different types of `Attention` and `MatrixAttention`
and observe the shapes of the input and the output. 

<codeblock source="part2/common-architectures/attention"></codeblock>

</exercise>


<exercise id="5" title="Common neural network techniques">

In addition to the modeling operations covered above, AllenNLP provides a variety of other
convenient abstractions and modules for building your NLP model.

## FeedForward

The [`FeedForward` module](http://docs.allennlp.org/master/api/modules/feedforward/) is just a
sequence of linear layers `torch.nn.Linear`. It also supports activations and (optional) dropout in
between layers.

## GatedSum and Highway layers

Linear combination of multiple components is a common operation used in modern-day neural networks.
A [highway layer](http://docs.allennlp.org/master/api/modules/highway/) computes a gated sum of the
original input `x` and its non-linear transformation: `y = g * x + (1 - g) * f(A(x))`, where `A()`
is a linear transformation and `f` is an element-wise nonlinearity. Highway layers are used e.g., in
ELMo after a CNN layer.

A somewhat similar operation, [gated sum](http://docs.allennlp.org/master/api/modules/gated_sum/)
computes a gated sum of two tensors `a` and `b` as in: `f = activation(W [a; b]) out = f * a + (1 -
f) * b`. 

## TimeDistributed

In some cases you may want to apply the same operation repeatedly over some dimension of a tensor.
For example, in a sequence labeling task, you might want to apply the same `FeedForward` layer to
the output vector at every timestep. 

[`TimeDistributed`](http://docs.allennlp.org/master/api/modules/time_distributed/) is a convenient
module that makes this easier to implement. Given an input of shape `(batch_size, time_steps,
[rest])`, and some `Module` that takes inputs of shape `(batch_size, [rest])`, the module unrolls
the second (time) dimension into the first (batch) dimension, reshaping it to be `(batch_size *
time_steps, [rest])`, applies the given module, and rolls it back to the original shape.

Besides being used for sequence labeling models, `TimeDistributed` can also be used to apply a
feedforward network to every span representation in span-based models (see above), to encode a list
of documents using a document-level encoder, or a host of other cases where a collection of things
should each have the same operation performed on them independently.

## Activations

AllenNLP abstracts activation functions as
[`Activations`](http://docs.allennlp.org/master/api/nn/activations/) so, e.g., `FeedForward` can
have a type to represent any possible activation. This is just a thin wrapper around PyTorch's
activation functions (such as `torch.nn.ReLU`). We use this type behind the scenes to instantiate
objects from configuration files; if you just want to instantiate an activation function, it's just
fine to create the pytorch class (like `torch.nn.ReLU`) directly yourself.  If you want, you can
also do:

```activation = Activation.by_name('relu')()```

The main place where we'd recommend using this type is as a constructor argument if you're creating
a module that's similar to `FeedForward`, or you really want to be able to easily swap out
activation functions in some place in your model.  Otherwise, you can safely ignore this type.

## Conditional random field

The (linear-chain) conditional random field (CRF) is a popular model used for sequence labeling in
NLP. It defines a globally-normalized conditional probability distribution over label sequences
given an input sequence by considering the dependency between labels (transition probabilities) and
between the input and the labels (emission probabilities).

AllenNLP's
[`ConditionalRandomField`](http://docs.allennlp.org/master/api/modules/conditional_random_field/)
implements a CRF layer, which, given a sequence of logits and another sequence of labels, computes
the log likelihood of the sequence. The module has transition probabilities as trainable parameters,
and can find the sequence of most likely tags using the Viterbi algorithm.

A CRF sequential tagger is implemented as the [`CrfTagger`
module](http://docs.allennlp.org/master/api/models/crf_tagger/). It takes a sequence of indexed
tokens, embeds them using a `TextFieldEmbedder`, encodes them with a `Seq2SeqEncoder`, applies a
`TimeDistributed` linear layer, and finally applies a `ConditionalRandomField` module.

</exercise>
