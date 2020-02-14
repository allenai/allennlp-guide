---
title: 'Reading textual data'
description: This chapter provides a deep dive into AllenNLP abstractions that are essential for reading textual data, including fields and instances, dataset readers, vocabulary, and how batching is handled in AllenNLP
type: chapter
---

<exercise id="1" title="Fields and instances">

## Fields

A `Field` contains one piece of data for one example that is passed through your model. `Fields` get converted to tensors in a model, either as an input or an output, after being converted to IDs and batched & padded. 

There are many types of fields in AllenNLP depending on the type of data they represent. Among them, the most important is `TextFields`, which represents a piece of tokenized text. [A chapter on representing text as features](/representing-text-as-features) will give you a deep dive into `TextFields` and related concepts.

Other commonly used fields include:

* `LabelField` is a categorical label. We used this for representing text labels in [Part 1 — quick start](/your-first-model).
* `MultiLabelField` is an extension of `LabelField` that allows for multiple labels. This can be used for, e.g., multilabel text classification.
* `SequenceLabelField` is a sequence of categorical labels. This can be used for, e.g., representing gold labels in sequential labeling tasks.
* `SpanField` is a pair of indices that represent a span of text. This can be used for, e.g., representing spans for reading comprehension, semantic role labeling, or coreference resolution.
* `ArrayField` is an array representing some data that you have already converted into a matrix, e.g., images and hand-crafted feature vectors

`Fields` can be created simply by supplying the data. `Field` objects provide APIs for creating empty fields, counting vocabulary items, creating tensors, and batching tensors, among others. See the following code snippet for more detail.

<codeblock source="part2/reading-textual-data/fields"></codeblock>

## Instances

An instance is the atomic unit of prediction in machine learning. In AllenNLP, `Instances` are collections of `Fields`, and datasets are collections of `Instances`.

`Instances` are created by dataset readers and used to create a `Vocabulary`. The `Vocabulary` is then used to map all strings in the `Instance`'s `Fields` into integer IDs, so that they can be turned into tensors. Later in the training pipeline, these tensors are batched together and fed to the model. The following diagram shows how `Fields` and `Instances` are created from a dataset.

<img src="/part2/reading-textual-data/fields-and-instances.svg" alt="Fields and Instances" />

`Instances` can be created by passing a dictionary of field names and corresponding fields to the constructor. `Instances` know how to turn themselves into a dictionary of field names and corresponding tensors, which is then used by `Batches` to batch together tensors of the same type. See the following code snippet for how to create instances and use their APIs.

The fields names are important—because the resulting dictionary of tensors is passed by name to the model after being destructured, they have to match the model's `forward()` arguments exactly.

<codeblock source="part2/reading-textual-data/instances"></codeblock>

</exercise>

<exercise id="2" title="Dataset readers">

## Basics of dataset readers

We already gave a quick intro to `DatasetReaders` and wrote our own in [Your first model](/your-first-model). `DatasetReaders` read datasets and convert them to collections of `Instances`. Here, we'll give a more in-depth look into what's going on inside a `DatasetReader`, what extra functionality it has, and why it works the way it does.

There are dataset readers available for a wide range of NLP tasks, including:

* [`TextClassificationJsonReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/text_classification_json.py) (for text classification)
* [`SequenceTaggingDatasetReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/sequence_tagging.py) (for sequence labeling)
* [`SimpleLanguageModelingDatasetReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/simple_language_modeling.py) (for language modeling)
* [`SnliReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/snli.py) (for NLI)
* [`SrlReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/semantic_role_labeling.py) (for span detection)
* [`Seq2SeqDatasetReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py) (for seq2seq models)
* [`PennTreeBankConstituencySpanDatasetReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/penn_tree_bank.py) and  [`UniversalDependenciesDatasetReader`](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/universal_dependencies.py) (for constituency and dependency parsing)
* Many [reading comprehension](https://github.com/allenai/allennlp-reading-comprehension) dataset readers such as [`SquadReader`](https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/dataset_readers/squad.py) and [`DropReader`](https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/dataset_readers/drop.py)
* Many [semantic parsing](https://github.com/allenai/allennlp-semparse) dataset readers such as [`AtisDatasetReader`](https://github.com/allenai/allennlp-semparse/blob/master/allennlp_semparse/dataset_readers/atis.py) and [`WikiTablesDatasetReader`](https://github.com/allenai/allennlp-semparse/blob/master/allennlp_semparse/dataset_readers/wikitables.py)

You can implement your own dataset reader by subclassing the `DatsetReader` class. The code snippet below is the dataset reader we implemented in [Your first model](/your-first-model). The returned dataset is a list of `Instances` by default.

<codeblock source="part2/reading-textual-data/dataset_reader_basic"></codeblock>

It is recommended that you separate out the logic for creating instances as the `text_to_instances()` method. As we mentioned in [Training and prediction](/training-and-prediction), by sharing a common logic between the training and the prediction pipelines, we are making the system less susceptible to any issues arising from possible discrepancies in how instances are created between the two, and making it very easy to put up a demo of your model. You can use the method as follows in, for example, your `Predictor`:

```python
reader = ClassificationTsvReader()
tokens = [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')]
label = 'pos'
instance = reader.text_to_instance(tokens, label)
```

`DatasetReaders` have two methods—`_read()` and `read()`. `_read()` is defined as an abstract method, and you must override and implement your own when building a `DatasetReader` subclass for your dataset. `read()` is the main method called from clients of the dataset reader. It implements extra functionalities such as cashing and lazy loading, and calls `_read()` internally. Both methods return an iterable of `Instances`.

The main method, `read()`, takes a filename as its parameter. The reason why dataset readers are designed this way is that if you specify dataset-specific parameters when constructing a `DatasetReader`, then you can apply them to any files. You can also design a dataset reader that handles more complex data setups. For example, you can write one that takes a directory as its constructor parameter and takes a simple key such as `train` and `dev` as a parameter to `read()`. [`TriviaQaReader`](https://github.com/allenai/allennlp-reading-comprehension/blob/fa60af5736a22455d275e663d3dd1ecc838e400c/allennlp_rc/dataset_readers/triviaqa.py#L31-L35), for example, is designed to work this way.

Dataset readers are designed to read data from a local file, although in some cases you may want to read data from a URL. AllenNLP provides a utility method `cached_path` to support this. If a URL is passed to the method it will download the resource to a local file and return its path. If you want your dataset reader to support both local paths and URLs, you can wrap `file_path` using `cached_path` in your `_read()` method as follows:

```python
from allennlp.common.file_utils import cached_path

...

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as lines:
            for line in lines:
```

## Lazy mode

Dataset readers also support reading data in a lazy manner, where a  `DatasetReader` yields instances as needed rather than returning a list of all instances at once. This comes in handy when your dataset is too big to fit into memory or you want to start training your model immediately. The lazy mode can also be used if you want different behavior at each epoch, for example, in order to do some sort of sampling.

When `lazy=True` is passed to a dataset reader's constructor, its `read()` method returns a `LazyInstances` object (instead of a list of `Instances`), which is a wrapper and an iterator that calls `_read()` and produces instances when called.

<codeblock source="part2/reading-textual-data/dataset_reader_lazy"></codeblock>

# Caching dataset

Reading and preprocessing large datasets can take a very long time. `DatasetReaders` can cache datasets by serializing created instances and writing them to disk. The next time the same file is requested the instances are deseriarized from the disk instead of being created from the file.

<codeblock source="part2/reading-textual-data/dataset_reader_cache"></codeblock>

Instances are serialized by `jsonpickle` by default, although you can override this behavior if you want.

</exercise>

<exercise id="3" title="Vocabulary">

`Vocabulary` is an important component in the AllenNLP, touching on and used by many other abstractions and components. Simply put, `Vocabulary` manages mappings from strings to integer IDs. It is created from instances and used for converting textual data (such as tokens and labels) to integer IDs (and eventually to tensors).

`Vocabulary` manages different mappings using a concept called *namespaces*. Each namespace is a distinct mapping from strings to integers, so strings in different namespaces are treated separately. This allows you to have separate indices for, e.g., 'a' as a word and 'a' as a character, or 'chat' in English and 'chat' in French (which means 'cat' in English). See the diagram below for an illustration:

<img src="/part2/reading-textual-data/vocabulary.svg" alt="Vocabulary" />

There's an important distinction between namespaces: padded and non-padded namespaces. By default, namespaces are padded, meaning the mapping reserves indices for padding and out-of-vocabulary (OOV) tokens. This is useful for indexing tokens, where OOV tokens are common and padding is needed (the next section gives more details on how padding works in AllenNLP).

Non-padded namespaces, on the other hand, do not reserve indices for special tokens. This is more suitable for, e.g., class labels, where you don't need to worry about these. By default, namespaces ending in `"tags"` or `"labels"` are treated as non-padded, but you can modify this behavior by supplying a `non_padded_namespaces` parameter when creating a `Vocabulary`.

A common way to create a `Vocabulary` object is to pass a collection of `Instances` to the `from_instances` method. You can look up indices by tokens using the `get_token_index()` method. You can also do the inverse (looking up tokens by indices) using `get_token_from_index()`. 

<codeblock source="part2/reading-textual-data/vocabulary_creation"></codeblock>

When your dataset is too large, you may want to "prune" your vocabulary by setting a threshold and only retaining words that appear more than that threshold. You can achieve this by passing a `min_count` parameter, which specifies the minimum count tokens need to meet to be included per namespace.

<codeblock source="part2/reading-textual-data/vocabulary_count"></codeblock>

You can instantiate `Vocabulary` not just from a collection of instances but by other means. The class method `from_files` allows you to load a serialized `Vocabulary` from a directory. This is often the one created by the `dry-run` command [mentioned previously](/next-steps#2). You can also use `from_files_and_instances` to expand a pre-built vocabulary with new data. In practice, however, you rarely need to call these class methods yourself. Specify `"type": "from_files"` in the `vocabulary` section of your config file if you want to load from a directory, and specify `"type": "extend"` if you want to extend a pre-built vocabulary (which uses the `from_files_and_instances` class method). If the `"type"` key is not specified, `Vocabulary` is created from instances by default.

When you are just writing your dataset reader and the model, you rarely need to worry about how `Vocabulary` is built and managed. You never "see" the vocabulary in your dataset reader—it will be constructed behind the scenes by AllenNLP and used by the iterator to index the instances. If you're using a pretrained contextualizer, its pre-built vocabulary is typically added automatically for you. 

The final constructed vocab gets passed to the model automatically. In the model constructor, you can use the information from `Vocabulary` to initialize model parameters. For example, in the `SimpleClassifier` model we built in Part 1, the size of the `"labels"` namespace is used to initialize the final linear layer of the classifier:

<pre data-line="4,10" class="language-python line-numbers"><code>
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
        self.accuracy = CategoricalAccuracy()
</code></pre>

</exercise>

<exercise id="4" title="Iterators, batching, and padding">

* How iterating and batching works in AllenNLP
* How padding works in AllenNLP
* (A diagram showing how batching works)
* Some details of bucket iterator (sorting_keys, maximum_samples_per_batch, etc.)

</exercise>
