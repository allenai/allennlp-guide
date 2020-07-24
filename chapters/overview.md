---
title: 'Overview'
description:
  "This chapter will give an overview of AllenNLP, and will outline the main chapters of this guide"
type: chapter
---

<exercise id="1" title="What is AllenNLP">

AllenNLP is an open source library for building deep learning models for natural language
processing, developed by [the Allen Institute for Artificial Intelligence](https://allenai.org/). It
is built on top of PyTorch and is designed to support researchers, engineers, students, etc., who
wish to build high quality deep NLP models with ease. It provides high-level abstractions and APIs
for common components and models in modern NLP. It also provides an extensible framework that makes
it easy to run and manage NLP experiments.

In a nutshell, AllenNLP is
- a library with well-thought-out abstractions encapsulating the common data and model operations
  that are done in NLP research
- a commandline tool for training PyTorch models
- a collection of pre-trained models that you can use to make predictions
- a collection of readable reference implementations of common / recent NLP models
- an experiment framework for doing replicable science
- a way to demo your research
- open source and community driven

AllenNLP is used by a large number of organizations and research projects.

</exercise>

<exercise id="2" title="About this guide">

This guide is separated into three parts, with each having a separate audience in mind:

* In part 1, geared towards someone who is brand new to the library, we give you a quick
  walk-through of main AllenNLP concepts and features. We'll build a complete, working NLP model
  (a text classifier) along the way.
* In part 2, aimed at existing users of the library who want to understand how all of the parts work
  and why they are designed the way they are, we give in-depth tutorials on individual abstractions
  and features of AllenNLP.
* Finally, in part 3, we walk through using AllenNLP to build models for various NLP tasks.  This is
  intended for use, e.g., in conjunction with a university course, showing the practical code behind
  the theory that you might get in a lecture, and reinforcing concepts that were taught.

This guide contains plenty of code snippets and "hands-on" examples that you can modify and run
(powered by [Binder](https://mybinder.org/)). There is also [a companion github
repository](https://github.com/allenai/allennlp-guide-examples) that contains full code needed to
train and run the models in your environment.

The entire guide is written for AllenNLP version 1.0. AllenNLP 1.0+ is required to run the code
snippets and exercises.  If you're using an older version of AllenNLP, much of what's in this guide
will still apply, though you might want to look at our [github
history](https://github.com/allenai/allennlp/tree/v0.9.0/tutorials) to find older versions of some
tutorial content.

</exercise>

<exercise id="3" title="Prerequisites (i.e., things we won't teach you here)">

As a reader of this guide, you need to be familiar with the Python programming language. Some
familiarity with PyTorch is helpful too—AllenNLP is based on PyTorch and we don't typically explain
PyTorch concepts and APIs in detail, so you should be at least willing to look them up when
necessary.

You may also want to have a good understanding of modern deep NLP models and techniques and machine
learning topics—we might have a little bit of description of the NLP concepts as we go, but our goal
is not to teach you NLP, but to teach you how to do NLP using AllenNLP. However, you don't need any
prior knowledge of AllenNLP—the whole point of this guide is to provide onboarding of AllenNLP!

</exercise>

<exercise id="4" title="Table of contents">

## Part 1

* [Your first model](/your-first-model): Implementing a text classification model
* [Training and prediction](/training-and-prediction): Training and evaluating the model
* [Next steps](/next-steps): Some teasers for what else you can do after your first model

## Part 2

* [Reading data](/reading-data): DatasetReaders, Fields, and Instances
* [Building your model](/building-your-model): All about AllenNLP's `Model` class
* [Common architectures](/common-architectures): Abstractions designed to make model building easy
* [Representing text as features](/representing-text-as-features): TextFields, Tokenizers,
  TokenIndexers, and TokenEmbedders
* [Testing your code](/testing): (Coming soon) Recommendations and utilities for testing NLP code
* [Building a demo](/demos-and-predictors): (Coming soon) Predictors, and designing code with demos in mind
* [Using config files](/using-config-files): FromParams and Registrable
* [Writing your own script](/writing-your-own-script) as an entry point to AllenNLP (Coming soon)
* [Debugging your code](/debugging) using an IDE debugger

## Part 3

* [Coming soon!](/coming-soon): A list of what we're working on and how to let us know what you'd
  like to see.
* [Hyperparameter Optimization with Optuna](/hyperparameter-optimization-with-optuna):
  A basic tutorial for hyperparameter optimization using Optuna

</exercise>
