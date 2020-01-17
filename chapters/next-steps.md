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

To recap, the model first uses `embedder` (of type `TextFieldEmbedder`) to convert the text into an embedding, then uses `encoder` (of type `Seq2VecEncoder`) to convert it to a vector (or a set of vectors when batched). The beauty of how models are designed in AllenNLP is that they mostly define "what" should be done in terms of abstractions and data types without being explicit about "how" it should be done. In fact, both `TextFieldEmbedder` and `Seq2VecEncoder` are defined as abstract classes, meaning they only provide interfaces, not implementations.

This design allows you to be flexible and swap in any components as long as they conform to the same interface. For example, a `Seq2VecEncoder` can be any of: bag of embeddings (which we did in the last chapter), a CNN, an RNN, a Transformer encoder, etc, as long as the module takes a sequence of vectors and produces a single vector. You can even implement your own encoder.

As a quick example, let's see how we can use BERT as a pretrained contextualizer for the text classification model. As shown below, you need to make four changes to the config file, while no change is needed for the model you wrote in Python.

<pre data-line="7-10,11-16,24-29,31-35" class="language-js"><code class="language-js">local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
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
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 5
    }
}
</code></pre>

We won't go into the details of these changes here, but in a nutshell, you need to:
1. Use a `PretrainedTransformerTokenizer` (`"pretrained_transformer"`), which tokenizes the string into wordpieces and adds special tokens like `[CLS]` and `[SEP]`
2. Use a `PretrainedTransformerIndexer` (`"pretrained_transformer"`), which converts those wordpieces into ids using BERT's vocabulary
3. Replace the embedder layer with a `PretrainedTransformerEmbedder` (`"pretrained_transformer"`), which uses a pretrained BERT model to embed the tokens, returning the top layer from BERT
4. Replace the encoder with a `BertPooler` (`"bert_pooler"`), which adds another (pretrained) linear layer on top of the `[CLS]` token and returns the result

Also note that we switched the optimizer to use `AdamW` from HuggingFace's Transformers library.

The tokenizer and the embedder are thin wrappers around [HuggingFace's Transformers library](https://github.com/huggingface/transformers), so switching between different transformer architectures (BERT, RoBERTa, XLNet, etc.) is as simple as changing the `model_name` parameters in the config file.

You can find the full config file [in the example repo](https://github.com/allenai/allennlp-course-examples/tree/master/quick_start). To train the model, you can use the command:

```
$ allennlp train \
    my_text_classifier.jsonnet \
    --serialization-dir model-bert \
    --include-package my_text_classifier
```

You should see a validation accuracy ~ 0.900, which is a huge improvement over the bag-of-embeddings model we built in the previous chapter. We achieved all this without changing a single line of Python code. Using AllenNLP, it is straightforward to swap in and out components of your NLP model without modifying Python code.

</exercise>

<exercise id="2" title="More AllenNLP commands">

In the previous chapter, we introduced three AllenNLP commands—`train`, `evaluate`, and `predict`. These are the most commonly used ones, but there are some other commands and command-line tools that you may find useful.

We briefly talked about the structure of AllenNLP config files, but do you find it confusing knowing what should go in them? Try out the configuration wizard, which allows you to choose components (dataset reader, model, iterator, etc.) and specify their arguments using a graphical interface, and it auto-generates the config file. 

You can install and run the config wizard using the following commands:

```
pip install allennlp-server
allennlp-configure --include-package my_text_classifier
```

<img src="/next-steps/config-wizard.png" alt="Config wizard" />

For more details, read [the chapter](/using-config-files) on config files, registrable, and from_params.

Also, we've been training and running the models using AllenNLP commands, but if you want to just write your own Python script instead of using commands and config files, that's fine too; there's [a chapter](/writing-python-script) on the best ways of doing this in part 3.

There are also `allennlp dry-run`, which creates a vocabulary and shows statistics of the dataset etc. without actually training the model, and [`allentune`](https://github.com/allenai/allentune), a hyperparameter search tool for AllenNLP. 

</exercise>

<exercise id="3" title="Running a demo">

After training an NLP model, you may want to deploy it as a Web application or show a demo to your colleagues. AllenNLP comes with a  demo server that serves your model via a simple Web interface.

The simple demo server is also in a separate package, which you can install by:

```
pip install allennlp-server
```

Then you can spin up the server by running:

```
python allennlp-server/server_simple.py \
    --archive-path model/model.tar.gz \
    --predictor sentence_classifier \
    --field-name sentence
    --include-package my_text_classifier
```

Note that you need to specify the name of the field(s) to accept input. You can access `localhost:8000` in your browser to see the simple demo:

<img src="/next-steps/simple-demo.png" alt="Simple demo" />

It is also relatively easy to customize this demo if you know basic HTML, CSS, and JavaScript. By default, all the assets (HTML, JavasScript code, and CSS) are hard-coded in `server_simple.py`, a Python script to launch the demo. You can change where these assets are served from by specifying the `--static-dir` option when running the script. See [this tutorial by AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/getting_started/predicting_paper_venues/predicting_paper_venues_pt2.md) for more details on how to customize the demo.

</exercise>

<exercise id="4" title="Using GPUs">

Finally, as long as you are using standard AllenNLP components (models, modules, etc.), you rarely need to worry about making your model compatible with GPUs by manually moving model parameters and tensors between devices. In most cases, all you need to do is add the `cuda_device` option to your trainer specifying the ID of the GPU you want to use:

```
    "trainer": {
        ...
        "cuda_device": 0
        ...
    }
```

If you have multiple GPUs, you can do distributed training by specify a list of GPU IDs in a `"distributed"` section:

```
    "trainer": {
        ...
    },
    "distributed": {
        "cuda_devices": [0, 1, 2, 3]
    }
```

This will use PyTorch's DistributedDataParallel to aggregate losses and synchronize parameter updates across multiple GPUs. The speedup you get, however, might not be exactly proportional to the number of GPUs due to due to synchronization and overhead.

When you are evaluating and making predictions with your model, you can specify the `--cuda-device` option from the command line to make your model run on GPUs.

</exercise>
