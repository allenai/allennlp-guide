---
title: 'Representing text as features: Tokenizers, TextFields, and TextFieldEmbedders'
description:
  "A deep dive into AllenNLP's core abstraction: how exactly we represent textual inputs, both on the data side and the model side."
type: chapter
---

<exercise id="1" title="Language to features">

<img src="/part2/representing-text-as-features/overview.svg" alt="Language to features" />

The basic problem in using machine learning on language data is converting symbolic text into numerical features that can be used by machine learning algorithms. In this chapter we will focus on the predominant modern approach to solving this problem, using (perhaps contextualized) word vectors; for a more in-depth discussion of and motivation for this approach, see [this
overview paper](https://arxiv.org/abs/1902.06006).

The first part of representing a string of text as numerical features is splitting the text up into individual `Tokens` that will each get their own representation.

`Tokens` are then converted into numerical values using a `TokenIndexer` with a help of a `Vocabulary`.

Finally, each individual ID gets replaced by a vector representing that word in some abstract space. The idea here is that words that are "similar" to each other in some sense will be close in the vector space, and so will be treated similarly by the model.

There are a lot of options for converting words into vector representations, including:

* GloVe or word2vec embeddings
* Character CNNs
* POS tag embeddings
* Combination of GloVe and character CNNs
* wordpieces and BERT

For example, the following diagram illustrates the conversion steps using a combination of GloVe and character CNNs. Note that a single token is mapped to two different types of IDs, which then get converted to different embeddings. 

<img src="/part2/representing-text-as-features/glove-cnn.svg" alt="Combination of GloVe and character CNNs" />

## Main steps

There were three steps in converting language to features:

1. Text → Tokens
2. Tokens → Ids
3. Ids → Vectors

The first two steps are data processing, while the last step is a modeling operation. Of these three steps, only the last step has learnable parameters. There are certainly decisions to be made in the first two steps that affect model performance, but they do not typically have learnable parameters that you want to train with backpropagation on your final task.

This separation means that what originally looked like a simple problem (representing text as features) actually needs coordination between two very different pieces of code: a `DatasetReader`, that performs the first two steps, and a `Model`, that performs the last step.

## Core abstractions

1. Tokenizer (Text → Tokens)
2. TextField and TokenIndexer (Tokens → Ids)
3. TextFieldEmbedder (Ids → Vectors)

We want our `DatasetReader` and `Model` to not have to specify which of all of the many possible options we're using to represent text as features; if we did that we would need to change our code to run simple experiments. Instead, we introduce high-level abstractions that encapsulate these operations and write our code using the abstractions. Then, when we run our code, we can construct the objects with the particular versions of these abstractions that we want (in software engineering, this is called _dependency injection_).

In the following sections, we will examine the first two abstractions in detail, showing how we go from text to something that can be input to a PyTorch `Model`.

</exercise>

<exercise id="2" title="The data side: TextFields">

# Language → Features: the data processing side


# Core abstractions

1. Tokenizer (Text → Tokens)
2. TextField (Tokens → Ids)
3. TextFieldEmbedder (Ids → Vectors)

Notes: As a reminder, the three main steps that get us from text to features are tokenization,
representing each token as some kind of id, then embedding those ids into a vector space.  In
AllenNLP, the first two steps happen on the data processing side inside a `DatasetReader`, and
we'll examine them more closely here.


# Core abstractions: Tokenizer

Options:

- Characters ("AllenNLP is great" → ["A", "l", "l", "e", "n", "N", "L", "P", " ", "i", "s", " ",
  "g", "r", "e", "a", "t"])
- Wordpieces ("AllenNLP is great" → ["Allen", "#NLP", "is", "great"])
- Words ("AllenNLP is great" → ["AllenNLP", "is", "great"])

Notes: There are three primary ways that NLP models today split strings up into individual tokens:
into characters (which includes spaces!), wordpieces (which split apart some words), or words.  In
AllenNLP, the tokenization that you choose determines what objects will get assigned single vectors
in your model - if you want a single vector per word, you need to start with a word-level
tokenization, even if you want that vector to depend on character-level representations.  How we
construct the vector happens later.


# Core abstractions: TextField

```python
TextField(tokens: List[Token], token_indexers: Dict[str, TokenIndexer])

# This is quite simplified, but shows the main points.
class TokenIndexer:
    def count_vocab_items(token: Token)
    def tokens_to_indices(tokens: List[Token], vocab: Vocabulary) -> Array
```

Notes: A `TextField` takes a list of `Tokens` from a `Tokenizer` and represents each of them as an
array that can be converted into a vector by the model.  It does this by means of a collection of
`TokenIndexers`.

Each `TokenIndexer` knows how to convert a `Token` into a representation that can be encoded by a
corresponding piece of the model.  This could be just mapping the token to an index in some
vocabulary, or it could be breaking up the token into characters or wordpieces and representing the
token by a sequence of indexed characters.  You can specify any combination of `TokenIndexers` in a
`TextField`, so you can, e.g., combine GloVe vectors with character CNNs in your model.

This abstraction is best understood by going through some examples; keep going to get some hands on
experience with how this works.

</exercise>

<exercise id="3" title="A simple TextField example">

In this exercise you can get some hands-on experience with how `TextField` data processing works.
We'll tokenize text and convert it into arrays using a `TextField`, printing out what it looks like
as we go.  First just run the code block below as is and see what comes out, then try modifying it
however you like to see what happens.

After you've gotten a feel for what's going on in the given example, as an exercise, see if you can
switch from using a single integer id for each word to representing words by the sequence
of characters that make them up.

Notice as you're looking at the output that the words in the vocabulary all start with index 2.
This is because index 0 is reserved for padding (which you can see once you finish the exercise),
and index 1 is reserved for out of vocabulary (unknown) tokens.  If you change the text to include
tokens that you don't add to your vocabulary, you'll see they get a 1 from the
`SingleIdTokenIndexer`.

<codeblock id="part2/representing-text-as-features/data/simple">
You'll want to use a `TokenCharactersIndexer` instead of a `SingleIdTokenIndexer`, and you'll need
to update the vocabulary, also.
</codeblock>

This time, let's modify the same code, but to use characters as tokens.  Both of these cases use
characters as their base input, but if we do it this way, in the model we'll get a single vector
per _character_, instead of per _word_.

<codeblock id="part2/representing-text-as-features/data/simple2">
You'll want to modify the `Tokenizer` and the `Vocabulary`.
</codeblock>

</exercise>

<exercise id="4" title="Combining multiple TokenIndexers">

In some cases in NLP we want to use multiple separate methods to represent text as vectors, then
combine them to get a single representation.  For instance, we might want to use pre-trained [GloVe
vectors](https://nlp.stanford.edu/projects/glove/) along with a character convolution to handle
unseen words (this was the approach taken by the [Bidirectional Attention
Flow](https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/007ab5528b3bd310a80d553cccad4b78dc496b02)
model (BiDAF), one of the first successful models on the [Stanford Question Answering
Dataset](https://rajpurkar.github.io/SQuAD-explorer/)).  The `TextField` abstraction is built with
this in mind, allowing you specify _any number_ of `TokenIndexers` that will get their own entries
in the tensors that are produced by the `TextField`.

In the following code, we show the setup used in BiDAF.  Run it to see what the output looks like
(pre-trained GloVe vectors happen in the model, not here, and in a real setting AllenNLP
automatically constructs the vocabulary in a special way that's informed by what words you have
vectors for).  See if you can modify it to add a third kind of representation: part of speech tag
embeddings.  Remember that the "embeddings" happen in the model, not here; we just need to get
_indices_ for the part of speech tags that we see in the text.

<codeblock id="part2/representing-text-as-features/data/combined">
The `Tokenizer` needs to be modified to include part of speech tags, you need a third entry in the
`token_indexers` dictionary, and the vocabulary needs to include part of speech tags.
</codeblock>

</exercise>

<exercise id="5" title="Contextualized representations in TextFields">

Here we'll show how the data processing works for using ELMo and BERT with AllenNLP (and in
principle, other pre-trained contextualizers).  They are both just different combinations of
`Tokenizers` and `TokenIndexers`, along with `TokenEmbedders` on the model side, which we'll get to
later.  This lets you try your modeling ideas with very simple (and quick to evaluate) word
representations, then move to the more sophisticated (and time-consuming) contextualizers after
you've settled on a basic model architecture, _without changing your model code_.

The code below shows usage with ELMo, and if you click on "show solution", we've also included an
example with BERT.  You can run both of these to see what the data looks like.

<codeblock id="part2/representing-text-as-features/data/contextual">
</codeblock>

For ELMo, you can see that each token gets a long sequence of 50 characters, the majority of which
are `261`, which ELMo uses to denote padding.

For BERT, each token gets split into wordpieces, then encoded with a single id.  But because the
tokens get split into wordpieces and then modeled by BERT in a single sequence, there are
additional keys output by the `TokenIndexer` to help the model reconstruct a single vector per
input token: `bert_tokens-offsets` tells the model which indices in the wordpiece sequence
correspond to the beginning of tokens.  `bert_tokens-type-ids` tells the model which side of a
`[SEP]` token each wordpiece is on (see the BERT paper for more info on that), and `mask` is used
to recover which tokens (not wordpieces) correspond to padding tokens.  Also note that the
`PretrainedBertIndexer` will automatically add `[CLS]` and `[SEP]` tokens at the beginning and end
of your text, so you don't have to do that manually.  This is why `bert_tokens-offsets` starts at
1.

</exercise>

<exercise id="6" title="The model side: TextFieldEmbedders">

# Language → Features: the model side


# Core abstractions

1. Tokenizer (Text → Tokens)
2. TextField (Tokens → Ids)
3. TextFieldEmbedder (Ids → Vectors)

Notes: As a reminder, the three main steps that get us from text to features are tokenization,
representing each token as some kind of id, then embedding those ids into a vector space.  In
AllenNLP, the last step happens inside a `Model`, and we'll examine it more closely here.


# Core abstractions: TextFieldEmbedder

```python
class TextFieldEmbedder:
    def forward(text_field_input: Dict[str, torch.Tensor])

class TokenEmbedder:
    def forward(inputs: torch.Tensor)
```

Notes: AllenNLP's data processing code converts each `TextField` in an `Instance` into a `Dict[str,
torch.Tensor]`, where each entry in that dictionary corresponds to one of the `TokenIndexers` that
we talked about in the last slide.

A `TextFieldEmbedder` takes that output, embeds or encodes each of the tensors individually using a
`TokenEmbedder`, then combines them in some way (typically by concatenating them), ending up with a
single vector per `Token` that was originally passed to the `TextField`.  You can then take those
vectors and do whatever modeling with them afterward that you want - the input text has been fully
converted into features that can be used in any machine learning algorithm.


# TokenIndexer → TokenEmbedders

- SingleIdTokenIndexer → Embedding
- TokenCharactersIndexer → TokenCharactersEncoder
- ElmoTokenIndexer → ElmoTokenEmbedder
- PretrainedBertIndexer → BertTokenEmbedder

Notes: Each `TokenIndexer` that you can use when constructing a `TextField` in your `DatasetReader`
has a corresponding `TokenEmbedder` that you use in your `Model` to encode the tensors created by
the `TokenIndexer`.  There are sometimes a few options for mixing and matching `Indexers` with
`Embedders`, but generally there is a one-to-one correspondence between these.  You have to make
sure that the collection of `TokenEmbedders` that you give to a `TextFieldEmbedder` in your model
matches the collection of `TokenIndexers` that you passed to the `TextField`.  This is typically
done in a [configuration file](/chapter14).

</exercise>

<exercise id="7" title="Embedding simple TextField inputs">

In this exercise we'll use `TextFieldEmbedders` to get a vector for each token inside a `Model`.
Below you'll see the code for doing this in the simple case where you have a single id representing
each token, and you just want to embed the token.  As an exercise, try converting this into using a
character-level CNN, to match the exercise we did above in changing the data processing to use a
`TokenCharactersIndexer`.

<codeblock id="part2/representing-text-as-features/model/simple">
We gave you the imports to use at the top of the file.  `CnnEncoder` takes three arguments:
`embedding_dim: int`, `num_filters: int`, and `ngram_filter_sizes: List[int]`.
`TokenCharactersEncoder` takes two arguments: `embedding: Embedding` and `encoder: Seq2VecEncoder`
(of which `CnnEncoder` is one).
</codeblock>

Notice that in both cases, your model typically will just be given a `TextFieldEmbedder` - all the
`Model` has to worry about is that it uses the `TextFieldEmbedder` to get a vector for each token,
and it doesn't have to care about how exactly that happens.  As we'll say repeatedly throughout
this course, this is a very important software design consideration that allows for cleaner and
more modular code.  It also helps you think at a higher level about the important parts of your
model as you're writing your model code.

</exercise>

<exercise id="8" title="Embedding text that has multiple TokenIndexers">

Here we'll parallel what we did above with using multiple `TokenIndexers`, showing how the model
changes to match.  We'll start with using two separate inputs, one that has single word ids, and
one that has a sequence of characters per token.  Notice how at the end, we end up with one vector
per input token, with both ways of encoding the tokens concatenated together.

As an exercise, see if you can modify this to take a third input that has a different single id,
corresponding to part of speech tags.  If you click on "show solution", we'll show you how the
token and part of speech tag vocabularies are actually used in practice to construct the embedding
layers with the right number of embeddings.

<codeblock id="part2/representing-text-as-features/model/combined">
You'll need to add another entry in the `token_tensor` dictionary, and a corresponding entry in the
`token_embedders` dictionary passed to the `BasicTextFieldEmbedder`.
</codeblock>

As we noted in the comments in the solution we provided, you'll see that putting together the data
and model side of `TextFields` in AllenNLP requires coordinating the keys used in a few different
places: (1) the vocabulary namespaces used by the `TokenIndexers` and the `TokenEmbedders` need to
match (where applicable), so that you get the right number of embeddings for each kind of input,
and (2) the keys used for the `TokenIndexer` dictionary in the `TextField` need to match the keys
used for the `TokenEmbedder` dictionary in the `BasicTextFieldEmbedder`.  We try to catch
mismatches here and give helpful error messages, but if you mismatch keys in either of these
places, you'll have problems in your code.

</exercise>

<exercise id="9" title="Using pretrained embeddings">
Below we give a simple example showing how to use pretrained embedding vectors (like GloVe) with
AllenNLP.  It's easiest to accomplish this with `Embedding.from_params()`.

<codeblock id="part2/representing-text-as-features/model/pretrained">
</codeblock>

</exercise>

<exercise id="10" title="Embedding contextualized inputs">

We'll finish this chapter with an example of how AllenNLP uses ELMo and BERT to produce a single
vector per token.  In these cases, the `TokenEmbedder` is actually operating on the entire sequence
of tokens, instead of just each token individually, doing some contextualization _inside_ the
`TextFieldEmbedder`.  But from your model's perspective, you still just have a `TextFieldEmbedder`
that gives you a single vector per token in your input; nothing changes in your code.

The original code shows ELMo, and if you click on "show solution" you'll see the BERT code.
Because we don't have a toy BERT model, you can't actually run the BERT code; it'll try to download
the huge BERT model, and it will time out.

<codeblock id="part2/representing-text-as-features/model/contextual">
</codeblock>

</exercise>
