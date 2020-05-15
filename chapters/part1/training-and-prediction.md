---
title: 'Training and prediction'
description:
  "This chapter will outline how to train your model and run prediction on new data"
type: chapter
---

<textblock>

In the previous chapter, we learned how to write your own dataset reader and model.  In this
chapter, we are going to train the text classification model and make predictions for new inputs.
At this point, there are two ways to proceed: you can write your own script to construct the dataset
reader and model and run the training loop, or you can write a configuration file and use the
`allennlp train` command.  We will show you how to use both ways here.

</textblock>

<exercise id="1" title="Training the model with your own script">

In this section we'll put together a simple example of reading in data, feeding it to the model,
and training the model, using your own python script instead of `allennlp train`.  While we
recommend using `allennlp train` for most use cases, it's easier to understand the introduction to
the training loop in this section.  Once you get a handle on this, switching to using our built in
command should be easy, if you want to.

Before proceeding, here are a few words about the dataset we will use throughout this chapter. The
dataset is derived from the [Movie Review
Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/), collections of movie reviews on
IMDb along with their polarity. The labels are binary (positive and negative), and our task is to
predict the label from the review text.

This section is going to give a series of executable examples, that you can run yourself in your
browser and see what they output.  They will build on each other, with code from previous examples
ending up in the `Setup` block in subsequent examples.

## Testing your dataset reader

In the first example, we'll simply instantiate the dataset reader, read the movie review dataset
using it, and inspect the AllenNLP `Instances` produced by the dataset reader.  Below we have code
that you can run (and modify if you want).

<codeblock source="part1/training-and-prediction/dataset_reader_source" setup="part1/training-and-prediction/dataset_reader_setup"></codeblock>

When you run the code snippet above, you should see the dumps of the first ten instances and their
content, including their text and label fields.  (Note that we are only showing the first 64 tokens
per instance by specifying `max_tokens=64`).

This is one way to check if your dataset reader is working as expected. We strongly recommend
[writing some simple tests](/testing) for your data processing code, to be sure it's actually doing
what you want it to.

## Feeding instances to the model

In the next example, we are going to instantiate the model and feed batches of instances to it.  At
this point, we're going to start breaking up our training loop into simple functions that
instantiate objects given their dependencies.  The `Model` needs to have a `Vocabulary` computed
from data before we can build it, but we don't really want to put the details of our model
construction inside our training loop function.  So to keep things sane, we'll pull out the model
building into a separate function that we call inside the main training function.

When you run this, you should see the outputs returned from the model. Each returned dict includes
the `loss` key as well as the `probs` key, which contains probabilities for each label.

<codeblock source="part1/training-and-prediction/model_source" setup="part1/training-and-prediction/model_setup"></codeblock>

## Training the model

Finally, we'll run backpropagation and train the model. AllenNLP uses a `Trainer` for this, which is
responsible for connecting necessary components (including your model, optimizer, instances, data
loader, etc.) and executing the training loop.

When you run this, the `Trainer` goes over the training data five times (`num_epochs=5`). After
each epoch, AllenNLP runs your model against the validation set to monitor how well (or badly) it's
doing. This is useful if you want to do, e.g., early stopping, and for monitoring in general.
Observe that the training loss decreases gradually—this is a sign that your model and the training
pipeline are doing what they are supposed to do (that is, to minimize the loss).

<codeblock source="part1/training-and-prediction/training_source" setup="part1/training-and-prediction/training_setup"></codeblock>

Congratulations, you just trained your first model using AllenNLP!  You probably don't want to
actually wait for that to finish running in binder, though; it's extremely slow. If you want, you
can head on over to our [course repository](https://github.com/allenai/allennlp-course-examples) and
run the code from there on your local machine.  Just run `python quick_start/train.py`; it finishes
in less than a minute on a Macbook.

</exercise>


<exercise id="2" title="Training the model with allennlp train">

Ok, we've seen how to set up a simple training loop.  Almost all of the training loop code above
that we wrote was in `build_*` functions, just constructing objects.  The `run_training_loop`
function itself was just about 10 lines of code.  But there are some details in that function, and
in the `build_*` functions, that are really important to get right, and it's mostly just boilerplate
that isn't really important to think about most of the time.  Additionally, we didn't add some nice
functionality, like having separate data loaders for validation or saving the trained model, that
add even more boilerplate to those methods.

We have a built-in training script that handles all of these things for you and makes it so the only
code that you have to write are your `DatasetReader` and `Model` classes.  Instead of writing all of
the `build_*` methods that we had above, we write a JSON configuration file specifying all necessary
parameters.  Our training script takes those parameters, creates all of the objects in the right
order, and runs the training loop.

There is an entire [chapter of this course](/using-config-files) dedicated to describing how the
configuration files work; here we'll just give a quick introduction to how to use them.  If you
decide you prefer writing your own script instead of using these configuration files, that's ok too,
and we have another [chapter of the course](/writing-your-own-script) that gives pointers on various
ways to make this easier.

## Configuration files

In a nutshell, configuration files in allennlp just take constructor parameters for various objects
and put them into a JSON dictionary.  Above, we had a `build_model` method that looked like this:

```python
def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)
```

This gets converted into a JSON dictionary that looks like this:

```javascript
"model": {
    "type": "simple_classifier",
    "embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 10
            }
        }
    },
    "encoder": {
        "type": "bag_of_embeddings",
        "embedding_dim": 10
    }
}
```

The constructor parameters to all of the objects that were created in `build_model` are translated
directly to keys in this dictionary.  AllenNLP relies on the type annotations in the model's
constructor code in order to construct these objects correctly.

There are two special things to note: first, to select a particular subclass of a base type (e.g.,
`SimpleClassifier` as a subclass of `Model`, or `BagOfEmbeddingsEncoder` as a subclass of
`Seq2VecEncoder`) we need an additional `"type": "simple_classifier"` key.  The string
`"simple_classifier"` comes from the call to `Model.register` that we saw in [the previous
chapter](/your-first-model#6)

Second, the `vocab` argument is missing here.  That's for the same reason that `vocab` was an
argument to the `build_model` method, not constructed inside it—the vocabulary gets constructed
separately, based on data, then passed in to the model.  Generally, the sequential dependencies
between objects that show up as arguments to your `build_*` methods are left out of the
configuration file, as they are handled in a different way.  Again, there's a lot more detail on how
this works in the [chapter on configuration files](/using-config-files).

We do this not just for the model, but for the dataset reader, the data loaders, the trainer, and
everything else that goes into a training loop.  This gives us a single JSON file that holds all of
the configuration for an experiment that was run (we actually use a superset of JSON called
[Jsonnet](https://jsonnet.org/), which supports fancier features like variables and imports, but a
plain JSON file works too).

For our simple classifier, that configuration file looks like this:

```js
{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
```

You can see entries there for all of the things we had `build_*` methods for (except the vocabulary,
which we omit because we are just using default parameters for that).  The configuration file is
read by matching keys in the JSON object by name with constructor parameters.  For the training
loop, the object we're constructing is called `TrainModel`, and you can see its constructor
[here](http://docs.allennlp.org/master/api/commands/train/#from_partial_objects).  The keys here
must exactly match those parameters, otherwise you get a `ConfigurationError`.

With this configuration file, we can train the model by running `allennlp train [config.json] -s
[serialization_directory]` from a command line.  In order for your dataset reader, model, and other
custom components to be recognized by the `allennlp` command, the calls to `.register()` have to be
run, which happens when the classes are imported.  So you typically have to also add the flag
`--include-package [my_python_module]`, or use allennlp's plugin functionality, when you run this
command.  There is more detail on how this works in the [chapter on configuration
files](/using-config-files).

Though you wouldn't typically run the code this way, we're including an example where we use the
configuration file, so you can play around with it if you want.

<codeblock source="part1/training-and-prediction/config_source" setup="part1/training-and-prediction/config_setup"></codeblock>

Again, you probably don't want to wait for that to finish, but you can run this on your local
machine by checking out our [course
repository](https://github.com/allenai/allennlp-course-examples/tree/master/quick_start).  `cd` to
the `quick_start` directory, then run `allennlp train my_text_classifier.jsonnet -s model
--include-package my_text_classifier`

There is definitely some overhead in getting used to these configuration files and understanding how
they work. We think they are useful enough to be worth the learning curve, but if you disagree, you
can still use all of the components of allennlp without them, as shown in the previous section.

</exercise>


<exercise id="3" title="Evaluating the model">

In this section, we will be evaluating the text classification model we just trained above, by
computing the evaluation metrics against the test set.

## Defining the metrics

In AllenNLP, you implement the logic to compute the metrics in your `Model` class. AllenNLP includes
an abstraction called `Metric` that gives some useful functionality for tracking metrics during
training. Here, we'll be using an accuracy `Metric`, `CategoricalAccuracy`, which computes the
fraction of instances for which our model predicted the label correctly.

First, you need to create an instance of `CategoricalAccuracy` in your model constructor:

<pre data-line="11" class="language-python line-numbers"><code>
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

Then, for each forward pass, you need to update the metric by feeding the prediction and the gold
labels:

<pre data-line="17" class="language-python line-numbers"><code>
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
        self.accuracy(logits, label)
        return {'loss': loss, 'probs': probs}
</code></pre>

The way metrics work in AllenNLP is that, behind the scenes, each `Metric` instance holds "counts"
that are necessary and sufficient to compute the metric. For accuracy, these counts are the number
of total predictions as well as the number of correct predictions. These counts get updated after
every call to the instance itself, i.e., the `self.accuracy(logits, label)` line. You can pull out
the computed metric by calling `get_metrics()` with a flag specifying whether to reset the counts.
This allows you to compute the metric over the entire training or validation dataset.

Finally, you need to implement the `get_metrics()` method in your model, which returns a dictionary
from metric names to their values, as shown below:

```python
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
```

AllenNLP's default training loop will call this method at the appropriate times and provide logging
information with current metric values.

## Evaluating the model

Now that your model is ready to compute the metric, let's run the entire pipeline where we train the
model, make the prediction and compute the metric against the test set.

In the code snippet below, we are reading a test set using the dataset reader, and using AllenNLP's
utility function `evaluate()` to run your model and get the metric on the test set. Notice that
AllenNLP now shows values for the defined metric as well as the the loss after each batch during
training and validation.

<codeblock source="part1/training-and-prediction/evaluation_source" setup="part1/training-and-prediction/evaluation_setup"></codeblock>

When this code snippet finishes running, you should see the evaluation result:

```
{'accuracy': 0.855, 'loss': 0.3686505307257175}
```

Though, once again, this is pretty slow to be running in binder.  You can run this from the [course
repository](https://github.com/allenai/allennlp-course-examples) with `python
quick_start/evaluate.py`.

As a simple bag-of-embeddings model, this is not a bad start!

## From command line

In order to evaluate your model from command line, you can use the `allennlp evaluate` command. This
command takes the path to the model archive file created by the `allennlp train` command, along with
the path to a file containing test instances, and returns the computed metrics.

</exercise>


<exercise id="4" title="Making predictions for unlabeled inputs">

In this final exercise, we'll be making predictions for new, unlabeled inputs using the trained text
classification model.

## Modifying the dataset reader

Before we move on to making predictions for unlabeled inputs, we need to make one change to the
dataset reader we wrote previously. Specifically, we refactor out the logic for creating an
`Instance` from tokens and the label as the `text_to_instance()` method as shown below.

We are making this piece of code sharable between `DatasetReader` and `Predictor` (which we'll
discuss in detail below). Why is this a good idea? Here, we are practically building two pipelines,
i.e., two different "flows" of data—one for training and another for prediction. By factoring out
the common logic for creating instances and sharing it between two pipelines, we are making the
system less susceptible to any issues arising from possible discrepancies in how instances are
created between the two, a problem known as [training-serving
skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew). This
may not seem too big of a deal in this tiny example, but making the feature extraction code sharable
between different pipelines is critical in real-world ML applications, and makes it very easy to put
up a demo, such as our [AllenNLP demos](https://demo.allennlp.org).

Note that we are making the `label` parameter of `text_to_instance()` optional. During training and
evaluation, all the instances were labeled, i.e., they included the `LabelFields` that contain gold
labels. However, when you are making predictions for unseen inputs, the instances are *unlabeled*.
By making the `label` parameter optional the dataset reader can support both cases.

<pre data-line="11-16" class="language-python line-numbers"><code>
@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                yield self.text_to_instance(text, sentiment)
</code></pre>

## Modifying the model

We also need to make some changes to the `forward()` method of our model. When we were training the
model, all the instances included the labels and the model was trained based on the loss computed
using those labels. However, during prediction, the instances the model gets are not labeled. The
`forward()` method doesn't need to (in fact, it can't) compute the loss—it just needs to return the
prediction.

In order to support prediction, first you need to make the `label` parameter optional by specifying
a default value of `None`. This will let you feed unlabeled instances to the model. Second, you need
to compute the loss and accuracy only when the label is supplied. See the modified version of
`forward()` below:

<pre data-line="4,16-19" class="language-python line-numbers"><code>
class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
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
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            # Shape: (1,)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output
</code></pre>

## Writing your predictor

For making predictions in a demo setting, AllenNLP uses `Predictors`, which are a thin wrapper
around your trained model. A `Predictor`'s main job is to take a JSON representation of an instance,
convert it to an `Instance` using the dataset reader (the `text_to_instance` mentioned above), pass
it through the model, and return the prediction in a JSON serializable format.

In order to build a `Predictor` for your task, you only need to inherit from `Predictor` and
implement a few methods (see `predict()` and `_json_to_instances()` below)—the rest will be taken
care of by the base class.

```python
@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)
```

AllenNLP provides implementations of `Predictors` for common tasks. In fact, it includes
`TextClassifierPredictor`, a generic `Predictor` for text classification tasks, so you don't even
need to write your own! Here, we are writing one from scratch solely for demonstration, but you
should always check whether the predictor for your task is already there.

## Making predictions

Now, let's put all the pieces together and make predictions for some unseen data.  In the code
snippet below, you are first training your model as we did above, then wrapping the model with a
`SentenceClassifierPredictor` to make predictions for new instances. Because the returned result
(`output['probs']`) is just an array of probabilities for class labels, we use
`vocab.get_token_from_index()` to convert a label ID back to its label string.

<codeblock source="part1/training-and-prediction/prediction_source" setup="part1/training-and-prediction/prediction_setup"></codeblock>

When you run the code above, you should get results similar to:

```
[('neg', 0.48853254318237305), ('pos', 0.511467456817627)]
[('neg', 0.5346643924713135), ('pos', 0.4653356373310089)]
```

This means that at least for these instances your model is making correct predictions!

## From command line

If you prefer interacting with models from a command line, AllenNLP provides a `predict` command to
make predictions.  This command is very similar to `allennlp evaluate` (see above)—it takes the path
to the model archive file created by the `allennlp train` command, along with the path to a JSON
file containing serialized test instances, and runs the model against these instances.

Overall, assuming that the model, dataset reader, and predictor we built so far are defined in some
module named `my_text_classifier`, you would use the following AllenNLP commands to train the model,
evaluate it, and make predictions for unseen data. Remember that you need to supply
`--include-package` option so that AllenNLP can find your module. All the example code is set up [in
this repo](https://github.com/allenai/allennlp-course-examples). You just need to clone it, cd to
the `quick_start` directory, and run these commands just like this was your own project directory.

```
$ allennlp train \
    my_text_classifier.jsonnet \
    --serialization-dir model \
    --include-package my_text_classifier

$ allennlp evaluate \
    model/model.tar.gz \
    data/movie_review/test.tsv \
    --include-package my_text_classifier

$ allennlp predict \
    model/model.tar.gz \
    data/movie_review/test.jsonl \
    --include-package my_text_classifier \
    --predictor sentence_classifier
```

And that's it! Though simple, we trained and ran a full-fledged NLP model using AllenNLP in this
chapter. In the next chapter we'll give a preview of some more advanced AllenNLP features, and
things you might want to try next.

</exercise>
