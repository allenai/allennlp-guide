---
title: 'Representing text as features: Tokenizers, TextFields, and TextFieldEmbedders'
description:
  "A deep dive into AllenNLP's core abstraction: how exactly we represent textual inputs, both on the data side and the model side."
type: chapter
---

<exercise id="1" title="Language to features">

<img src="/part2/representing-text-as-features/overview.svg" alt="Language to features" />

The basic problem in using machine learning on language data is converting symbolic text into
numerical features that can be used by machine learning algorithms. In this chapter we will focus on
the predominant modern approach to solving this problem, using (perhaps contextualized) word
vectors; for a more in-depth discussion of and motivation for this approach, see [this overview
paper](https://arxiv.org/abs/1902.06006).

The first part of representing a string of text as numerical features is splitting the text up into
individual `Tokens` that will each get their own representation.

`Tokens` are then converted into numerical values using a `TokenIndexer` with a help of a
`Vocabulary`.

Finally, each individual ID gets replaced by a vector representing that word in some abstract space.
The idea here is that words that are "similar" to each other in some sense will be close in the
vector space, and so will be treated similarly by the model.

There are a lot of options for converting words into vector representations, including:

* GloVe or word2vec embeddings
* Character CNNs
* POS tag embeddings
* Combination of GloVe and character CNNs
* wordpieces and BERT

For example, the following diagram illustrates the conversion steps using a combination of GloVe and
character CNNs. Note that a single token is mapped to two different types of IDs, which then get
converted to different embeddings.

<img src="/part2/representing-text-as-features/glove-cnn.svg" alt="Combination of GloVe and character CNNs" />

## Main steps

There were three steps in converting language to features:

1. Text → Tokens
2. Tokens → Ids
3. Ids → Vectors

The first two steps are data processing, while the last step is a modeling operation. Of these three
steps, only the last step has learnable parameters. There are certainly decisions to be made in the
first two steps that affect model performance, but they do not typically have learnable parameters
that you want to train with backpropagation on your final task.

This separation means that what originally looked like a simple problem (representing text as
features) actually needs coordination between two very different pieces of code: a `DatasetReader`,
that performs the first two steps, and a `Model`, that performs the last step.

## Core abstractions

In AllenNLP, we take each of the three main steps above and build an abstraction around it:

1. Tokenizer (Text → Tokens)
2. TextField, TokenIndexer, and Vocabulary (Tokens → Ids)
3. TextFieldEmbedder (Ids → Vectors)

We want our `DatasetReader` and `Model` to not have to specify which of all of the many possible
options we're using to represent text as features; if we did that we would need to change our code
to run simple experiments. Instead, we introduce high-level abstractions that encapsulate these
operations and write our code using the abstractions. Then, when we run our code, we can construct
the objects with the particular versions of these abstractions that we want (in software
engineering, this is called [_dependency injection_](/using-config-files#1)).

In the following sections, we will examine these abstractions in detail, showing first how we go
from text to something that can be input to a PyTorch `Model`, then how the `Model` uses those
inputs to produce embeddings.

</exercise>


<exercise id="2" title="Tokenizers and TextFields">

## Tokenizers

`Tokenizers` split strings up into individual tokens. There are three primary ways to do this in
AllenNLP:

- Characters ("AllenNLP is great" → `["A", "l", "l", "e", "n", "N", "L", "P", " ", "i", "s", " ",
  "g", "r", "e", "a", "t"]`)
- Wordpieces ("AllenNLP is great" → `["Allen", "##NL", "##P", "is", "great"]`)
- Words ("AllenNLP is great" → `["AllenNLP", "is", "great"]`)

Note that characters include whitespace too. Wordpieces are similar to words, but further split
words into subword units.

In AllenNLP, the tokenization that you choose determines what objects will get assigned single
vectors in your model—if it's important that you have a single vector per word (e.g., so you can
predict an NER or POS tag for it), you need to start with a word-level tokenization, even if you
want that vector to depend on character-level representations. How we construct the vector happens
later.

Commonly used `Tokenizers` include `SpacyTokenizer`, which uses spaCy's tokenizer to split strings
into tokens and is available in many languages, and `PretrainedTransformerTokenizer`, which uses a
tokenizer from Hugging Face's `transformers` library.  Another popular choice is
`CharacterTokenizer`, which splits a string up into individual characters, including whitespace.

`Tokenizers` all implement a `tokenize()` method, which returns a list of `Tokens`. A `Token` is a
lightweight `dataclass` that is similar to spaCy's `Token`.

## TextFields

A `TextField` takes a list of `Tokens` from a `Tokenizer` and represents each of them as an array
that can be converted into a vector by the model. `TextFields` implement the [`Field`
API](/reading-data#1), which includes methods for counting vocabulary items, converting strings to
integers and then tensors, and batching together several tensors with proper padding.  To maintain
flexibility in how each of these are actually done, the `TextField` passes off most of the
responsibility for these operations to a collection of `TokenIndexers`, which actually does the job
of deciding how to represent each `Token`.  To create a `TextField`, you write code that looks
something like this:

```python
tokenizer = ...  # Whatever tokenizer you want
sentence = "We are learning about TextFields"
tokens = tokenizer.tokenize(sentence)
token_indexers = {...}  # we'll talk about this in the next section
text_field = TextField(tokens, token_indexers)
...
instance = Instance({"sentence": text_field, ...})
```

Passing off the responsibility of actually doing the processing to the `TokenIndexers` lets us have
the same code work with arbitrary collections of `TokenIndexers`, giving us a lot of flexibility
later, as we'll see.

</exercise>


<exercise id="3" title="TokenIndexers">

## TokenIndexers basics

Each `TokenIndexer` knows how to convert a `Token` into a representation that can be encoded by a
corresponding piece of the model. This could be just mapping the token to an index in some
vocabulary, or it could be breaking up the token into characters or wordpieces and representing the
token by a sequence of indexed characters. You can specify any combination of `TokenIndexers` in a
`TextField`, so you can, e.g., combine GloVe vectors with character CNNs in your model.

In the following code snippet, we'll tokenize text and convert it into tensors using a `TextField`,
printing out what it looks like as we go. First, we'll split the text into word tokens and index
them with single IDs using a `SingleIdTokenIndexer`. Next, we switch to representing each token with
a sequence of characters that make it up using a `TokenCharactersIndexer`. Finally, we use
characters as tokens, and index them using single IDs using a `SingleIdTokenIndexer`.

Notice as you're looking at the output that the words in the vocabulary all start with index 2. This
is because index 0 is reserved for padding (which you can see in the final tensor), and index 1 is
reserved for out of vocabulary (unknown) tokens.  If you change the text to include tokens that you
don't add to your vocabulary, you'll see they get a 1 from the `SingleIdTokenIndexer`.

<codeblock source="part2/representing-text-as-features/token_indexers_simple" setup="part2/representing-text-as-features/setup"></codeblock>

## Combining multiple TokenIndexers

In some cases in NLP we want to use multiple separate methods to represent text as vectors, then
combine them to get a single representation. For instance, we might want to use pre-trained [GloVe
vectors](https://nlp.stanford.edu/projects/glove/) along with a character convolution to handle
unseen words (this was the approach taken by the [Bidirectional Attention
Flow](https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/007ab5528b3bd310a80d553cccad4b78dc496b02)
(BiDAF) model, one of the first successful models on the [Stanford Question Answering
Dataset](https://rajpurkar.github.io/SQuAD-explorer/)). The `TextField` abstraction is built with
this in mind, allowing you specify _any number_ of `TokenIndexers` that will get their own entries
in the tensors that are produced by the `TextField`.

In the following code, we show the setup used in BiDAF—representing each token as a combination of
GloVe vectors and character embeddings.  We then add a third kind of representation: part of speech
tag embeddings. Remember that the "embeddings" happen in the model, not here; we just need to get
_indices_ for the words, characters, and part of speech tags that we see in the text.

<codeblock source="part2/representing-text-as-features/token_indexers_combined" setup="part2/representing-text-as-features/setup"></codeblock>

</exercise>


<exercise id="4" title="The model side: TextFieldEmbedders">

As a reminder, the three main steps that get us from text to features are:

1. Tokenization (Text → Tokens) using `Tokenizers`
2. Representing each token as some kind of ID (Tokens → Ids) using `TextFields` and `TokenIndexers`
3. Embedding those IDs into a vector space (Ids → Vectors) using `TextFieldEmbedders`

In AllenNLP, the last step happens inside a `Model`, and we'll examine it more closely here.

AllenNLP's data processing code converts each `TextField` in an `Instance` into a `TextFieldTensors`
data structure, which is just an alias of `Dict[str, Dict[str, torch.Tensor]]`. Each entry in the
outer dictionary corresponds to one of the `TokenIndexers` that we talked about in the last slide.
The inner dictionary contains the objects produced by a given `TokenIndexer`.

A `TextFieldEmbedder` takes that output, embeds or encodes each `TokenIndexer`'s output individually
using a `TokenEmbedder`, then combines them in some way (typically by concatenating them), ending up
with a single vector per `Token` that was originally passed to the `TextField`.  You can then take
those vectors and do whatever modeling with them afterward that you want - the input text has been
fully converted into features that can be used in any machine learning algorithm.

There are a number of details in how all of this works that are easier to explain through code.  Be
sure to carefully read the comments as you go through the following code, where we use
`TextFieldEmbedders` to get a vector for each token inside a `Model`. First, you'll see a simple
case where you have a single ID representing each token, and you just want to embed the token. Next,
you'll be using a character-level CNN, to match the exercise we did above in changing the data
processing to use a `TokenCharactersIndexer`.

<codeblock source="part2/representing-text-as-features/token_embedders_simple" setup="part2/representing-text-as-features/setup"></codeblock>

Our recommended coding style in AllenNLP is to have a model simply take a `TextFieldEmbedder` as a
constructor parameter.  This means that all the `Model` has to worry about is that it uses the
`TextFieldEmbedder` to get a vector for each token, and it doesn't have to care about how exactly
that happens. Because both of the cases above return a single vector per token, the rest of the
model doesn't care about how it is done, and both cases work just fine without changing any model
code.  As we'll say repeatedly throughout this guide, this is a very important software design
consideration that allows for cleaner and more modular code.  It also helps you think at a higher
level about the important parts of your model as you're writing your model code.


## Embedding text that has multiple TokenIndexers

Here we'll parallel what we did above with using multiple `TokenIndexers`, showing how the model
side changes to match. We'll start with using two separate inputs, one that has single word IDs, and
one that has a sequence of characters per token. Notice how at the end, we end up with one vector
per input token, with both ways of encoding the tokens concatenated together.

Next, we add a third input that has a different single ID, corresponding to part of speech tags.
We'll show you how the token and part of speech tag vocabularies are actually used in practice to
construct the embedding layers with the right number of embeddings.

<codeblock source="part2/representing-text-as-features/token_embedders_combined" setup="part2/representing-text-as-features/setup"></codeblock>

As we noted in the comments in the code above, you'll see that putting together the data and model
side of `TextFields` in AllenNLP requires coordinating the keys used in a few different places: (1)
the vocabulary namespaces used by the `TokenIndexers` and the `TokenEmbedders` need to match (where
applicable), so that you get the right number of embeddings for each kind of input, and (2) the keys
used for the `TokenIndexer` dictionary in the `TextField` need to match the keys used for the
`TokenEmbedder` dictionary in the `BasicTextFieldEmbedder`.

</exercise>


<exercise id="5" title="Coordinating the three parts">

Now that we've seen a full example of going through all three steps of the text-to-feature pipeline,
let's take a step back and talk about how each of these pieces has to fit together.  It's your job
as you configure your code to select which components to use as concrete `Tokenizers`,
`TokenIndexers`, and `TokenEmbedders`.  You need to be sure that you select components that fit
together, or your code will not work.  For example, it doesn't really make sense to choose a
`CharacterTokenizer` and a `TokenCharactersIndexer`, because that `Indexer` assumes that you've
tokenized into words.  Similarly, we pass the outputs of a `TokenIndexer` to its corresponding
`TokenEmbedder` *by name*, which means that not all `TokenEmbedders` are compatible with all
`TokenIndexers`.

Typically, there is a one-to-one relationship between a `TokenIndexer` and a `TokenEmbedder`, and
each `TokenIndexer` mostly only makes sense with one `Tokenizer`.  Below is a mostly-exhaustive list
of the options available in AllenNLP:

Using a word-level tokenizer (such as `SpacyTokenizer` or `WhitespaceTokenizer`):

- `SingleIdTokenIndexer` → `Embedding` (for things like GloVe or other simple embeddings, including
  learned POS tag embeddings)
- `TokenCharactersIndexer` → `TokenCharactersEncoder` (for things like a character CNN)
- `ElmoTokenIndexer` → `ElmoTokenEmbedder` (for ELMo)
- `PretrainedTransformerMismatchedIndexer` → `PretrainedTransformerMismatchedEmbedder` (for using a
  transformer like BERT when you really want to do modeling at the word level, e.g., for a tagging
  task; more on what this does [below](#7))

Using a character-level tokenizer (such as `CharacterTokenizer`):

- `SingleIdTokenIndexer` → `Embedding`


Using a wordpiece tokenizer (such as `PretrainedTransformerTokenizer`):

- `PretrainedTransformerIndexer` → `PretrainedTransformerEmbedder`
- `SingleIdTokenIndexer` → `Embedding` (if you don't want contextualized wordpieces for some reason)

</exercise>


<exercise id="6" title="Using pretrained contextualizers and embeddings">

## Contextualized representations in TextFields

Here we'll show how the text-to-features pipeline works for using ELMo, BERT and other pre-trained
contextualizers with AllenNLP.  Because of how we designed this pipeline, switching from a simple
embedding to a pre-trained contextualizer just means choosing a different combination of
`Tokenizers`, `TokenIndexers`, and `TokenEmbedders`. This lets you try your modeling ideas or write
your [model tests](/testing) with very simple (and quick to evaluate) word representations, then
move to the more sophisticated (and time-consuming) contextualizers after you've settled on a basic
model architecture, _without changing your data or model code_, only its configuration.

We'll look at the data side of using contextualized representations first.  The following example
shows how ELMo and BERT `Tokenizers` and `TokenIndexers` work.

<codeblock source="part2/representing-text-as-features/token_indexers_contextual" setup="part2/representing-text-as-features/setup"></codeblock>

The important thing to notice is that the implementation code to actually tokenize text and create
the `TextFields` is identical to what we saw above.  The only thing that's different is the concrete
objects we chose to instantiate as `Tokenizers` and `TokenIndexers`.  The one exception to this is
at the end, when we called `tokenizer.add_special_tokens()` on two separate pieces of text, to do
joint modeling with a transformer.  `add_special_tokens` is a method that's specific to
`PretrainedTransformerTokenizer`.  If you are modeling pairs of text, because the way this is done
with and without transformers is so different, you currently have to have code that is specific to
transformers.

We'll finish this section with an example of how AllenNLP uses ELMo and BERT to produce a single
vector per token.  In these cases, the `TokenEmbedder` is actually operating on the entire sequence
of tokens, instead of just each token individually, doing some contextualization _inside_ the
`TextFieldEmbedder`.  But from your model's perspective, you still just have a `TextFieldEmbedder`
that gives you a single vector per token in your input; nothing changes in your code.

<codeblock source="part2/representing-text-as-features/token_embedders_contextual" setup="part2/representing-text-as-features/setup"></codeblock>

## Pretrained embeddings

Using pretrained (non-contextualized) embeddings, like
[GloVe](https://www.semanticscholar.org/paper/Glove%3A-Global-Vectors-for-Word-Representation-Pennington-Socher/f37e1b62a767a307c046404ca96bc140b3e68cb5),
is as simple as passing a path to the pretrained file to the `Embedding` class.  `Embedding` will
take your `Vocabulary` object and add vectors for every token found in both the `Vocabulary` and the
pretrained file.  Tokens not found in the pretrained file will get random embeddings, initialized
with the same mean and variance that was in the vectors in the pretrained file.  If you want to have
the `Vocabulary` take into account the pretrained file when deciding which words to keep, there are
constructor parameters available in `Vocabulary` for that, as well.

<codeblock source="part2/representing-text-as-features/pretrained_embedding"></codeblock>

</exercise>


<exercise id="7" title="Doing word-level modeling with a wordpiece transformer">

As mentioned above, the three main steps in going from text to features in a model are:

1. Text → Tokens
2. Tokens → Ids
3. Ids → Vectors

These three steps all have to match, and in particular, it's a hard constraint in the design of
AllenNLP that a `TextFieldEmbedder` has to produce one vector for each `Token` that is passed to a
`TextField`.  There's a case where this becomes tricky: what should you do if you want to compute a
loss, or predict labels, based on _word_ tokens, but you want to use a transformer (or some other
model) that operates on _subword_ units?

We have this problem in part-of-speech tagging, or named entity recognition.  These datasets are
annotated at the word level, so we should ideally model them at the word level.  There are two main
options for handling this case: either you do some pooling over wordpieces after running your
transformer, or you do distribute your labels (and your loss function) to the wordpiece level.

### Pooling over wordpieces

To better understand what's going on in this case, we'll go back to the three main steps.  The first
step is tokenization, and here we tokenize at the word level (typically the tokenization will be
already given to you, so you don't need to run a tokenizer at all).  In the second step (indexing),
we need to further tokenize each word into subword units, getting a list of wordpieces that will be
indexed and passed to the transformer in the third step (embedding).  The embedding step has to run
the transformer, then perform pooling to undo the subword tokenization that was done in the indexing
subwordtep, so that we end up with one vector per original token.  This is similar to how a
character-level CNN breaks words up into characters, then encodes them back to get a single vector
per word.

AllenNLP includes a `TokenIndexer` and `TokenEmbedder` that perform these operations For any
transformer in Hugging Face's `transformers` library.  They are the
`PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`.  We show an
example of their usage in the example below.  Notice how the original tokenization has no special
tokens; the `Indexer` adds them, does wordpiece tokenization, then flattens the wordpieces into a
single list, while providing a mapping to recover the original tokens (with special tokens removed)
in the `Embedder`.  The pooling operation done by the embedder is just a simple average of the
wordpiece vectors, though this could be easily modified if desired, by copying the code to your own
`Embedder` and changing it to perform the pooling operation that you want.

<codeblock source="part2/representing-text-as-features/mismatched_tokenization" setup="part2/representing-text-as-features/setup"></codeblock>

### Distributing labels from words to wordpieces

AllenNLP does not currently have any utilities implemented for this case, but it is relatively
straightforward to write a function to take word labels and convert them into wordpiece labels for
any sequence tagging task.  Once this is done, a standard `PretrainedTransformerIndexer` and
`PretrainedTransformerEmbedder` are sufficient to do modeling on the wordpiece-labeled data.

Another option is to give empty labels for non-initial wordpieces, and mask out the loss computation
for any wordpieces with empty labels.  This is problematic from a modeling perspective, however, as
it breaks CRF locality assumptions (you will get nothing useful from CRF transition probabilities)
and makes modeling generally harder.  We don't recommend this approach.

For non-tagging tasks where inputs are typically pre-tokenized (such as coreference resolution), we
don't have any specific advice other than to think carefully about which case makes the most sense
from a modeling perspective, and write your code accordingly.

</exercise>


<exercise id="8" title="How padding and masking works">

In NLP, we typically deal with variable-length sequences, because sentences or other textual inputs
to a model aren't all the same length.  But in order to make efficient use of GPUs, we need to do
batched computation, which requires tensors with fixed lengths.  To make these fixed-length tensors
for a given batch of inputs, we _pad_ the short sequences so that they are the same length as the
longest sequence in the batch.[^1]

[^1]: See the [`BucketBatchSampler`](/reading-data#4) for AllenNLP's built-in way to organize batches
so as to minimize the amount of padding required.

This padding is all done for you in AllenNLP's data processing code.  In the code examples
throughout this chapter, we have seen calls to `text_field.get_padding_lengths()`. If you look at
the output of that method, you'll see all of the tensor dimensions which might need padding, with
values assigned for a single input.  These dimensions are things like the number of tokens in a
`TextField`, the number of characters in a token, or the number of wordpieces output by a tokenizer.
Our `collate_function`, which PyTorch `DataLoaders` use to batch together inputs, finds the max
value for each padding dimension across a batch, then passes those maximum values to
`text_field.as_tensor()`, so that each individual tensor is created with a fixed length before
batching is performed.

Padding gives us fixed length tensors for a given batch, but it means that we have portions of our
input that are actually empty values.  For many modeling operations, such as attention
distributions, we need to know which values are actually empty, so that we don't assign probability
mass to those tokens (or spans, or whatever the padding is representing).  The masking computation
that is necessary depends on your model; AllenNLP handles some of it inside of a
`TextFieldEmbedder`, but in other places you need to make sure your model code does the proper
masking computation.  To help with this, AllenNLP provides several utility functions, some of which
are described in the next section.  In addition to what's described there, many functions in
`allennlp.nn.util` provide masked versions of PyTorch utility functions, such as `masked_softmax`,
`masked_log_softmax`, and `masked_topk`.

</exercise>


<exercise id="9" title="Interacting with TextField outputs in your model code">

If you use a `TextField` in your data processing code with AllenNLP, you will get a
`TextFieldTensors` object back, which is a complex dictionary structure, as we've seen in all of the
examples above.  The nice thing about this structure is that it is flexible, allowing you to change
the underlying representation without changing your model code.  But that flexibility can also be a
hinderance to usability sometimes, if you really want to pull out certain pieces of what's inside
the object.

We strongly recommend against writing code that directly accesses the internals of a
`TextFieldTensors` object.  When you do this, you hard-code assumptions about what representation
you used, and you make it very difficult to change those representations later.

Instead, AllenNLP provides several utilities for processing and accessing what's inside the
`TextFieldTensors`.  The first of these is the `TextFieldEmbedder` object, which we've seen before.
This converts the `TextFieldTensors` objet into one embedded vector per input token.  The other
common operations that you might want to do with a `TextFieldTensors` object are to get a mask out
of it for use in modeling operations, and to get the token ids, either for use in some
language-modeling-like task, or for displaying them to a human after converting them back to
strings.  `allennlp.nn.util` provides utility functions for both of these operations,
`get_text_field_mask` and `get_token_ids_from_text_field_tensors`, which are demonstrated in the
code below.

<codeblock source="part2/representing-text-as-features/interacting_with_tensors" setup="part2/representing-text-as-features/setup"></codeblock>

</exercise>
