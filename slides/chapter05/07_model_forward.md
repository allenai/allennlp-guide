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
        embedded_text = self.embedder(text)
        mask = util.get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)
        logits = self.classifier(encoded_text)
        probs = torch.nn.functional.softmax(logits)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}
```

