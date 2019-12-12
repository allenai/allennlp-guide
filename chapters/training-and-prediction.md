---
title: 'Training and prediction'
description:
  "This chapter will outline how to train your model and run prediction on new data"
prev: /your-first-model
next: /next-steps
type: chapter
id: 203
---

<textblock>

In the previous chapter, we learned how to write your own dataset reader and model, and how config files work in AllenNLP. In this chapter, we are going to train the text classification model and make predictions for new inputs.

</textblock>

<exercise id="1" title="Training the model">

In this exercise we'll put together a simple example of reading in data, feeding it to the model, and training the model.

Before proceeding, here are a few words about the dataset we will use throughout this chapter—they are derived from the [Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/), collections of movie reviews on IMDb along with their polarity. The labels are binary (positive and negative), and our task is to predict the label from the review text.

## Testing your dataset reader

In the first example, we'll simply instantiate the dataset reader, read the movie review dataset using it, and inspect the AllenNLP `Instances` produced by the dataset reader:

<codeblock source="training-and-prediction/dataset_reader" setup="training-and-prediction/setup"></codeblock>

Note that we are using a utility method called `run_config()`, which parses a given config file (here, it's just a JSON string), instantiates components such as the dataset reader and the model, and runs the training loop (if `trainer` is specified). In practice, you'll be writing AllenNLP commands such as `allennlp train` and `allennlp predict` in the terminal. We'll discuss AllenNLP commands later.

When you run the code snippet above, you should see the dumps of the first ten instances and their content, including their text and label fields. This is a great way to check if your dataset reader is working as expected.

## Feeding instances to the model

In the next example, we are going to instantiate the model and feed batches of instances to it. Note that the config file now has the `model` section, which contains the full specification for how to instantiate your model along with its sub-components. Also notice the `iterator` section in your config, which specifies how a `DataIterator` is instantiated and how instances are turned into batches. Here we are using a `BasicIterator` (`type: "basic"`), which simply creates fixed-sized batches.

<codeblock source="training-and-prediction/model" setup="training-and-prediction/setup"></codeblock>

When you run this, you should see the outputs returned from the model. Each returned dict includes the `loss` key as well as the `probs` key, which contains probabilities for each label. 

## Training the model

Finally, we'll run backpropagation and train the model. AllenNLP uses `Trainer` for this (specified by the `trainer` section in the config file), which is responsible for connecting necessary components (including your model, instances, iterator, etc.) and executing the training loop. It also includes a section for the optimizer we use for training. We are using an Adam optimizer with its default parameters.

<codeblock source="training-and-prediction/training" setup="training-and-prediction/setup"></codeblock>

When you run this, the `Trainer` goes over the training data five times (`"num_epochs": 5`). After each epoch, AllenNLP runs your model against the validation set to monitor how well (or badly) it's doing. This is useful if you want to do e.g., early stopping, and for monitoring in general. Observe that the training loss decreases gradually—this is a sign that your model and the training pipeline are doing what they are supposed to do (that is, to minimize the loss).

In practice, you'd be running AllenNLP commands from your terminal. For training a model, you'd do:

```
allennlp train [config.jsonnet] --serialization-dir model
```

where `[config.jsonnet]` is the config file and `--serialization-dir` specifies where to save your trained model.

Congratulations, you just trained your first model using AllenNLP!

</exercise>

<exercise id="2" title="Making predictions for new inputs">

</exercise>

<exercise id="3" title="Evaluating the model">

</exercise>
