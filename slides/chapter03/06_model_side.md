---
type: slides
---

# Language → Features: the model side

---

# Core abstractions

1. Tokenizer (Text → Tokens)
2. TextField (Tokens → Ids)
3. TextFieldEmbedder (Ids → Vectors)

Notes: As a reminder, the three main steps that get us from text to features are tokenization,
representing each token as some kind of id, then embedding those ids into a vector space.  In
AllenNLP, the last step happens inside a `Model`, and we'll examine it more closely here.

---

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
