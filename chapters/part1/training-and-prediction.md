---
title: 'Training and prediction'
description:
  "This chapter will outline how to train your model and run prediction on new data"
type: chapter
---

<textblock>

In the previous chapter, we learned how to write your own dataset reader and model, and how config files work in AllenNLP. In this chapter, we are going to train the text classification model and make predictions for new inputs.

</textblock>

<exercise id="1" title="Training the model">

In this exercise we'll put together a simple example of reading in data, feeding it to the model, and training the model.

Before proceeding, here are a few words about the dataset we will use throughout this chapter. The dataset is derived from the [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/), collections of movie reviews on IMDb along with their polarity. The labels are binary (positive and negative), and our task is to predict the label from the review text.

## Testing your dataset reader

In the first example, we'll simply instantiate the dataset reader, read the movie review dataset using it, and inspect the AllenNLP `Instances` produced by the dataset reader.

Below we have code that you can run (and modify if you want). We are using a utility method called `run_config()`, which parses a given config file (here, it's just a JSON string), instantiates components such as the dataset reader and the model, and runs the training loop (if `trainer` is specified). It also returns these components but we are not using them here.

In practice, you'll be using AllenNLP commands such as `allennlp train` and `allennlp predict` in the terminal. We'll discuss AllenNLP commands later.

<codeblock source="training-and-prediction/dataset_reader" setup="training-and-prediction/setup"></codeblock>

When you run the code snippet above, you should see the dumps of the first ten instances and their content, including their text and label fields. This is a great way to check if your dataset reader is working as expected. (Note that we are only showing the first 64 tokens per instance by specifying `"max_tokens": 64`).

## Feeding instances to the model

In the next example, we are going to instantiate the model and feed batches of instances to it. Note that the config file now has the `model` section, which contains the full specification for how to instantiate your model along with its sub-components. Also notice the `iterator` section in your config, which specifies how to batch instances together before passing them to your model. We go into more detail on how this works in [a chapter on reading textual data](/reading-textual-data).

When you run this, you should see the outputs returned from the model. Each returned dict includes the `loss` key as well as the `probs` key, which contains probabilities for each label. 

<codeblock source="training-and-prediction/model" setup="training-and-prediction/setup"></codeblock>

## Training the model

Finally, we'll run backpropagation and train the model. AllenNLP uses a `Trainer` for this (specified by the `trainer` section in the config file), which is responsible for connecting necessary components (including your model, instances, iterator, etc.) and executing the training loop. It also includes a section for the optimizer we use for training. We are using an Adam optimizer with its default parameters.

When you run this, the `Trainer` goes over the training data five times (`"num_epochs": 5`). After each epoch, AllenNLP runs your model against the validation set to monitor how well (or badly) it's doing. This is useful if you want to do, e.g., early stopping, and for monitoring in general. Observe that the training loss decreases gradually—this is a sign that your model and the training pipeline are doing what they are supposed to do (that is, to minimize the loss).

<codeblock source="training-and-prediction/training" setup="training-and-prediction/setup"></codeblock>

In practice, you'd be running AllenNLP commands from your terminal. For training a model, you'd do:

```
allennlp train [config.jsonnet] --serialization-dir model
```

where `[config.jsonnet]` is the config file and `--serialization-dir` specifies where to save your trained model. When the training finishes, this command will package up necessary files (e.g., the model weights and the vocabulary) into a single archive file `model.tar.gz`.

Congratulations, you just trained your first model using AllenNLP!

</exercise>

<exercise id="2" title="Evaluating the model">

In this section, we will be evaluating the text classification model we just trained above, by computing the evaluation metrics against the test set.

## Defining the metrics

In AllenNLP, you implement the logic to compute the metrics in your `Model` class. AllenNLP includes an abstraction called `Metric` that represents various metrics. Here, we'll be simply using accuracy (`CategoricalAccuracy`), the fraction of instances for which our model predicted the label correctly.

First, you need to create an instance of `CategoricalAccuracy` in your model constructor:

<pre data-line="11" class="language-python line-numbers"><code>class SimpleClassifier(Model):
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

Then, for each forward pass, you need to update the metric by feeding the prediction and the gold labels:

<pre data-line="17" class="language-python line-numbers"><code>class SimpleClassifier(Model):
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

The way metrics work in AllenNLP is that, behind the scenes, each `Metric` instance holds "counts" that are necessary and sufficient to compute the metric. For accuracy, these counts are the number of total predictions as well as the number of correct predictions. These counts get updated after every call to the instance itself, i.e., `accuracy()`. You can pull out the computed metric by calling `get_metrics()` with a flag specifying whether to reset the counts. This allows you to compute the metric over the entire training or validation dataset.

Finally, you need to implement the `get_metrics()` method in your model, which returns a dictionary from metric names to their values, as shown below:

```python
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
```

AllenNLP's default training loop will call this method at the appropriate times and provide logging information with current metric values.

## Evaluating the model

Now that your model is ready to compute the metric, let's run the entire pipeline where we train the model, make the prediction and compute the metric against the test set.

In the code snippet below, we are reading a test set using the dataset reader, and using AllenNLP's utility function `evaluate()` to run your model and get the metric on the test set. Notice that AllenNLP now shows values for the defined metric as well as the the loss after each batch during training and validation.

<codeblock source="training-and-prediction/evaluation" setup="training-and-prediction/setup"></codeblock>

When this code snippet finishes running, you should see the evaluation result:

```
{'accuracy': 0.855, 'loss': 0.3686505307257175}
```

As a simple bag-of-embeddings model, this is not a bad start!

## From command line

In order to evaluate your model from command line, you can use the `allennlp evaluate` command. This command takes the path to the model archive file created by the `allennlp train` command, along with the path to a file containing test instances, and returns the computed metrics.

</exercise>

<exercise id="3" title="Making predictions for unlabeled inputs">

In this final exercise, we'll be making predictions for new, unlabeled inputs using the trained text classification model.

## Modifying the dataset reader

Before we move on to making predictions for unlabeled inputs, we need to make one change to the dataset reader we wrote previously. Specifically, we refactor out the logic for creating an `Instance` from tokens and the label as the `text_to_instance()` method as shown below.

We are making this piece of code sharable between `DatasetReader` and `Predictor` (which we'll discuss in detail below). Why is this a good idea? Here, we are practically building two pipelines, i.e., two different "flows" of data—one for training and another for prediction. By factoring out a common logic for creating instances and sharing it between two pipelines, we are making the system less susceptible to any issues arising from possible discrepancies in how instances are created between the two, a problem known as [training-serving skew](https://developers.google.com/machine-learning/guides/rules-of-ml#training-serving_skew). This may not seem too big of a deal in this tiny example, but making the feature extraction code sharable between different pipelines is critical in real-world ML applications.

Note that we are making the `label` parameter of `text_to_instance()` optional. During training and evaluation, all the instances were labeled, i.e., they included the `LabelFields` that contain gold labels. However, when you are making predictions for unseen inputs, the instances are *unlabeled* and do not contain the labels. By making the `label` parameter optional the dataset reader can support both cases. 

<pre data-line="11-16" class="language-python line-numbers"><code>@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], label: str = None) -> Instance:
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)
                yield self.text_to_instance(tokens, sentiment)
</code></pre>

## Modifying the model

We also need to make some changes to the `forward()` method of our model. When we were training the model, all the instances included the labels and the model was trained based on the loss computed using those labels. However, during prediction, the instances the model gets are not labeled. The `forward()` method doesn't need to (in fact, it can't) compute the loss—it just needs to return the prediction.

In order to support prediction, first you need to make the `label` parameter optional by specifying its default value (`None`). This will let you feed unlabeled instances to the model. Second, you need to compute the loss only when the label is supplied. See the modified version of `forward()` below:

<pre data-line="4,16-19" class="language-python line-numbers"><code>class SimpleClassifier(Model):
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
        # Shape: (1,)
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output
</code></pre>

## Writing your predictor

For making predictions, AllenNLP uses `Predictors`, which are a thin wrapper around your trained model. A `Predictor`'s main job is to take a JSON representation of an instance, convert it to an `Instance` using the dataset reader (the `text_to_instance` mentioned above), pass it through the model, and return the prediction in a JSON serializable format.

In order to build a `Predictor` for your task, you only need to inherit from `Predictor` and implement a few methods (see `predict()` and `_json_to_instances()` below)—the rest will be taken care of by the base class. 

```python
@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer()

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)
```

AllenNLP provides implementations of `Predictors` for common tasks. In fact, it includes `TextClassifierPredictor`, a generic `Predictor` for text classification tasks, so you don't even need to write your own! Here, we are writing one from scratch solely for demonstration, but you should always check whether the predictor for your task is already there.

## Making predictions

Now, let's put all the pieces together and make predictions for some unseen data.  In the code snippet below, you are first training your model using the same configuration as the one we used in the previous section, then wrapping the model with a `SentenceClassifierPredictor` to make predictions for new instances. Because the returned result (`output['probs']`) is just an array of probabilities for class labels, we use `vocab.get_token_from_index()` to convert a label ID back to its label string.

<codeblock source="training-and-prediction/prediction" setup="training-and-prediction/setup"></codeblock>

When you run the code above, you should get results similar to:

```
[('neg', 0.48853254318237305), ('pos', 0.511467456817627)]
[('neg', 0.5346643924713135), ('pos', 0.4653356373310089)]
```

This means that at least for these instances your model is making correct predictions!

## From command line

In practice, instead of writing code to feed data to a predictor, you'll be using the `allennlp predict` command to make predictions. This command is very similar to `allennlp evaluate` (see above)—it takes the path to the model archive file created by the `allennlp train` command, along with the path to a JSON file containing serialized test instances, and runs the model against these instances.

Overall, assuming that the model, dataset reader, and predictor we built so far are defined in some module named `my_text_classifier`, you would use the following AllenNLP commands to first train the model, evaluate it, and make predictions for unseen data. Remember that you need to supply `--include-package` option so that AllenNLP can find your module. All the example code is set up [in this repo](https://github.com/allenai/allennlp-course-examples). You just need to clone it, cd to the `quick_start` directory, and run these commands just like this was your own project directory.

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

And that's it! Though simple, we trained and ran a full-fledged NLP model using AllenNLP in this chapter. In the next chapter we'll touch upon some more AllenNLP features.

</exercise>
