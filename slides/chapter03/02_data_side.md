---
type: slides
---

# Language → Features: the data processing side

---

# Core abstractions

1. Tokenizer (Text → Tokens)
2. TextField (Tokens → Ids)
3. TextFieldEmbedder (Ids → Vectors)

Notes: As a reminder, the three main steps that get us from text to features are tokenization,
representing each token as some kind of id, then embedding those ids into a vector space.  In
AllenNLP, the first two steps happen on the data processing side inside a `DatasetReader`, and
we'll examine them more closely here.

---

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

---

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
