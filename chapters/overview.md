---
title: 'Course overview'
description:
  "This chapter will give an overview of AllenNLP, and will outline the main chapters of this course"
prev: null
next: /introduction
type: chapter
---

<exercise id="1" title="What is AllenNLP">

AllenNLP is an open source framework for research on deep learning methods in natural language processing, developed by [the Allen Institute for Artificial Intelligence](https://allenai.org/). It is built on top of PyTorch and is designed to support researchers, engineers, students etc. who wish to build high quality deep NLP models with ease. It provides high-level abstractions and APIs for common components and models in modern NLP. It also provides an extensible framework that makes it easy to run and manage NLP experiments.

In a nutshell, AllenNLP is
- a library with well-thought-out abstractions encapsulating the common data and model operations
  that are done in NLP research
- a commandline tool for training PyTorch models
- a collection of pre-trained models that you can use to make predictions
- a collection of readable reference implementations of common / recent NLP models
- an experiment framework for doing replicable science
- a way to demo your research
- open source and community driven

As of this writing, AllenNLP is used by a number of organizations and research projects.

</exercise>

<exercise id="2" title="About this course">

This course provides onboarding of AllenNLP and in-depth tutorials on how to use the framework as well as its components for building NLP models and systems. This course consists of three main parts:

* Part 1 gives you a quick walk-through of main AllenNLP concepts and features. We'll build a complete, working NLP model (text classifier) along the way.
* Part 2 provides in-depth tutorials on individual abstractions and features of AllenNLP.
* Part 3 introduces common NLP tasks and how to build models for these tasks using AllenNLP.

This course contains plenty of code snippets and "hands-on" exercises that you can modify and run (powered by [Binder](https://mybinder.org/)). There is also [a companion repo](https://github.com/allenai/allennlp-course-examples) that contains full code needed to train and run the models in your environment. 

The entire course is written for AllenNLP version 1.0. AllenNLP 1.0+ is required to run the code snippets and exercises.

</exercise>

<exercise id="3" title="Target audience">

As a reader of this course, you need to be familiar with the Python programming language. Some familiarity with PyTorch is helpful too—AllenNLP is based on PyTorch and we don't typically explain PyTorch concepts and APIs in detail, so you should be at least willing to look them up when necessary.

You may also want to have a good understanding of modern deep NLP models and techniques and machine learning topics—we might have a little bit of description of the NLP concepts as we go, but our goal is not to teach you NLP, but to teach you how to do NLP using AllenNLP. However, you don't need any prior knowledge of AllenNLP—the whole point of this course is to provide onboarding of AllenNLP!

</exercise>

<exercise id="4" title="Table of contents">

## Part 1

* [Introduction](/introduction): what is text classification
* [Your first model](/your-first-model): implementing a text classification model
* [Training and prediction](/training-and-prediction): training and evaluating the model
* [Next steps](/next-steps): introducing pre-trained contextualizers and more AllenNLP features

## Part 2

* [Reading textual data](/reading-textual-data): DatasetReaders, Fields, and Instances
* Building your model
* Common architectures
* Representing text as features
* Training and evaluating your model
* Testing your code
* Using config files: Registrable and FromParams
* Building a demo
* Interpreting your model’s predictions
* Writing your own script as an entry point to AllenNLP

## Part 3

* Text classification
* Sequence labeling
* Language modeling
* Sentence pair classification
* Span classification
* Syntactic parsing
* Reading comprehension
* Sequence to sequence models
* Semantic parsing

</exercise>
