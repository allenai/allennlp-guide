---
type: slides
---

# Implementing the Model - the forward method

---

# Reminder: the constructor and our input/output spec

```python
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
```

```python
# Inputs:
text: TextField

# Outputs:
label: LabelField
```

Notes: This was the constructor from the last section, and our input/output spec from earlier.
Here we'll show how to use these parameters inside of `Model.forward()`, which will get arguments
that match our input/output spec (because that's how we coded the [`DatasetReader`](chapter05#3)).

---

# Model.forward()

```python
class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
```

Notes: In `forward`, we use the parameters that we created in our constructor to try to predict the
outputs from the inputs in our input/output spec.  After we've predicted the outputs, we compute
some loss function based on how close we got to the true outputs, and then return that loss (along
with whatever else we want) so that we can use it to train the parameters.

In the next few slides we'll again go over this line by line and explain what's going on.

---

# Model.forward()

<pre data-line="3-4"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

Notes: The first thing to notice is the inputs to this function.  The way the `Trainer` works in
AllenNLP is that we will take the field names that you used in your `DatasetReader` and give you a
batch of instances with _those same field names_ in `forward`.  So, because we used `text` and
`label` as our field names, we need to name our arguments to `forward` the same way.

Second, notice the types of these arguments.  Each type of `Field` knows how to convert itself into
a `torch.Tensor`, then create a batched `torch.Tensor` from all of the `Fields` with the same name
from a batch of `Instances`.  The types you see for `text` and `label` are the tensors produced by
`TextField` and `LabelField`.  We won't go into the details of why `TextField` produces a
`Dict[str, torch.Tensor]` here; see our [chapter on using TextFields](#) for more information about
that.  The important part to know is that our `TextFieldEmbedder`, which we created in the
constructor, expects this type of object as input and will return an embedded tensor as output.

---

# Model.forward()

<pre data-line="5-6"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

Notes: The first actual modeling operation that we do is embed the text, getting a vector for each
input token.  Notice here that we're not specifying anything about _how_ that operation is done,
just that a `TextFieldEmbedder` that we got in our constructor is going to do it.  This lets us be
very flexible later, changing between various different kinds of embedding methods or pretrained
representations (including ELMo and BERT) without changing our code.

---

# Model.forward()

<pre data-line="7-10"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

Notes: After we have embedded our text, we next have to squash the sequence of vectors (one per
token) into a single vector for the whole text.  We do that using the `Seq2VecEncoder` that we got
as a constructor argument.  In order to behave properly when we're batching pieces of text together
that could have different lengths, we need to _mask_ elements in the `embedded_text` tensor that
are only there due to padding.  We use a utility function to get a mask from the `TextField`
output, then pass that mask into the encoder.  At the end of these lines, we have a single vector
for each instance in the batch.

---

# Model.forward()

<pre data-line="11-16"><code class="language-python">class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

Notes: The last step of our model is to take the vector for each instance in the batch and predict
a label for it.  Our `classifier` is a `torch.nn.Linear` layer that gives a score (commonly called
a `logit`) for each possible label.  We normalize those scores to get a probability distribution
over labels that we can return to a consumer of this model.  For computing the loss, pytorch has a
built in function that computes the cross entropy between the logits that we predict and the true
label distribution, and we use that as our loss function.

And that's it!  This is all you need for a simple classifier.
