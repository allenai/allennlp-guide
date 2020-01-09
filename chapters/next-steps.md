---
title: 'Next steps'
description:
  "Now that you have a working model, here are some things you can try with AllenNLP!"
prev: /training-and-prediction
next: null
type: chapter
id: 204
---

<textblock>

In the previous two chapters, we were able to quickly build a working NLP model using AllenNLP, although we just scratched the surface of what the library has to offer so far. AllenNLP offers more models, modules, and features that make it easier to develop a wide range of NLP applications. In this chapter, we'll give you a preview of some more things you can try with AllenNLP, along with some pointers to later chapters that give you more detail on individual topics.

</textblock>

<exercise id="1" title="Switching to pre-trained contextualizers">

Now let's look at the definition of our text classifier model again. The following figure illustrates how the model (its forward method) processes the input and comes up with the label.

<pre class="language-python"><code class="language-python">class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
</code></pre>

<img src="/next-steps/pretrained-contextualizers.svg" alt="Using pre-trained contextualizers" />

To recap, the model first uses `embedder` (of type `TextFiledEmbedder`) to convert the text into an embedding, then uses `encoder` (of type `Seq2VecEncoder`) to convert it to a vector (or a set of vectors when batched). The beauty of how models are designed in AllenNLP is that they mostly define "what" should be done in terms of abstractions and data types without being explicit about "how" it should be done. In fact, both `TextFiledEmbedder` and `TextFields`  are defined as abstract classes, meaning they only provide interfaces, not implementations.

This design allows you to be flexible and swap in any components as long as they confirm the same interface. For example, a `Seq2VecEncoder` can be any of: bag of embeddings (which we did in the last chapter), a CNN, an RNN, a Transformer encoder, etc, as long as the module takes a sequence of vectors and produces a single vector. You can even implement your own encoder.

As a quick example, let's see how we can use BERT as a pretrained contexturalizer for the text classification model. As shown below, you need to make four changes to the config file, while no change is needed for the model you wrote in Python.

<pre data-line="7-9,11-14,27-32,36-38" class="language-js"><code class="language-js">local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "word_splitter": "bert-basic"
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
            }
        }
    },
    "train_data_path": "data/movie_review/train.tsv",
    "validation_data_path": "data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": bert_model,
                    "top_layer_only": true,
                    "requires_grad": false
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model,
            "requires_grad": false
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
</code></pre>

We won't go into the details of these changes here, but in a nutshell, you need to:
1. Use a `BertBasicWordSplitter` (`"bert-basic"`) for splitting words
2. Use a `PretrainedBertIndexer` (`"bert-pretrained"`) for indexing words (BERT uses word pieces internally)
3. Replace the `TokenEmbedder` with a `PretrainedBertEmbedder` (`"bert-pretrained"`) for embedding tokens
4. Replace the `Seq2VecEncoder` with a `BertPooler` (`"bert_pooler"`) for extracting the embedding for the `[CLS]` token

Also note that you need to add two (somewhat cryptic) fields to your embedder: `allow_unmatched_keys` and `embedder_to_indexer_map`. We'll cover in detail how to represent text in AllenNLP (including how `TextFields`, `TokenIndexers`, and `TokenEmbedders` work) in [a chapter in Part 3](/representing-text-as-features).

You can find the full config file [in the example repo](https://github.com/allenai/allennlp-course-examples/tree/master/quick_start). To train the model, you can use the command:

```
$ allennlp train \
    my_text_classifier.jsonnet \
    --serialization-dir model-bert \
    --include-package my_text_classifier
```

You should see a validation accuracy ~ 0.565, which is not a huge improvement over the bag-of-embeddings model we built in the previous chapter, but remember we haven't done any hyperparameter tuning whatsoever. The point of this exercise is to show you it is relatively straightforward to swap in and out components of your NLP model without changing a single line of Python code.

</exercise>

<exercise id="2" title="More AllenNLP commands">

* allennlp configure (config wizard)
* allennlp dry-run

</exercise>

<exercise id="3" title="Running a demo">

* Running a simple server demo
* Customizing the demo

</exercise>

<exercise id="4" title="Using GPUs and Docker">

* Using GPUs
* Using Docker

</exercise>
