---
title: 'Common architectures'
description: "In this chapter we'll introduce neural architectures and AllenNLP abstractions that are commonly used for building your NLP model."
type: chapter
---

<textblock>
The main modeling operations done on natural language inputs include summarizing sequences, contextualizing sequences (that is, computing contextualized embeddings from sequences), modeling spans within a longer sequence, and modeling similarities between sequences using attention. In the following sections we'll learn AllenNLP abstractions for these operations.
</textblock>

<exercise id="1" title="Summarizing sequences">

Taking a sequence of tokens and summarizing it to a fixed-size vector is one of the most fundamental operations done on natural language inputs. AllenNLP provides an abstraction called `Seq2VecEncoder` for this, which is a class of architectures that take a sequence of vectors and summarize it to a single vector of fixed size. This includes a wide range of models, from something very simple (a bag-of-embedding model which simply sums up the input embeddings) to something more complicated (a pooling layer of BERT). See the following diagram for an illustration:

<img src="/part2/common-architectures/seq2vec.svg" alt="Seq2Vec encoder" />

RNNs are a popular choice for summarizing sequences in many NLP models. Instead of implementing its own RNN-based `Seq2VecEncoders`, AllenNLP offers `PytorchSeq2VecWrapper`, which wraps PyTorch's existing RNN implementations (such as `torch.nn.LSTM` and `torch.nn.GRU`) and make them compatible with AllenNLP's `Seq2VecEncoder` interface. In most cases, though, you don't need to use the wrapper yourself—wrapped `Seq2VecEncoders` are already defined by AllenNLP as e.g., `LstmSeq2VecEncoder` and `GruSeq2VecEncoder`.

Other commonly used `Seq2VecEncoders` include `CnnEncoder`, which is a combinations of convolutional layers and a pooling layer on top of them, as well as `BertPooler`, which returns the embedding for the `[CLS]` token of the BERT model.

We already used a `Seq2VecEncoder` when we built [a sentiment classifier in Part 1](your-first-model#3). Remember that we defined the constructor of `SimpleClassifier` to take any `Seq2VecEncoder`:

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

This allows you to plug in any `Seq2VecEncoder` implementations when defining the model. For example, you can do:

```python
...
encoder = LstmSeq2VecEncoder(input_size=10, hidden_size=10, num_layers=1)
model = SimpleClassifier(vocab, embedder, encoder)
...
```

Your model will use a 1-layer LSTM with `input_size=10` and `hidden_size=10` for summarizing the input sequence.

</exercise>

<exercise id="2" title="Contextualizing sequences">

* Seq2SeqEncoder
    * RNN
* Sample code

</exercise>

<exercise id="3" title="Modeling spans in sequences">

* SpanField
* SpanExtractor
* Sample code
* pruning

</exercise>

<exercise id="4" title="Modeling similarities between sequences">

* Attention
* MatrixAttention
    * Why two abstractions for attention
* Sample code
    * Similarity matrix computation in BiDAF

</exercise>

<exercise id="5" title="Common neural network techniques">

* FeedForward
* Activations
    * Sample code
* ConditionalRandomField
* Highway and residual connection
* TimeDistributed
* GatedSum

</exercise>
