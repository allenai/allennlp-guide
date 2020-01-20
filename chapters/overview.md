---
title: 'Course overview'
description:
  "This chapter will give an overview of AllenNLP, and will outline the main chapters of this course"
prev: null
next: /introduction
type: chapter
id: 101
---

<exercise id="1" title="What is AllenNLP">

AllenNLP is an open source framework for research on deep learning methods in natural language processing, developed by [the Allen Institute for Artificial Intelligence](https://allenai.org/). It is built on top of PyTorch and is designed to support researchers, engineers, students etc. who wish to build high quality deep NLP models with ease. It provides high-level abstractions and APIs for common components and models in modern NLP. It also provides an extensible framework that makes it easy to run and manage NLP experiments.

As of this writing, AllenNLP is used by a number of organizations and research projects.

</exercise>

<exercise id="2" title="About this course">

This course provides onbording of AllenNLP and in-depth tutorials on how to use the framework as well as its components for building NLP models and systems. This course consists of three main parts:

* Part 1 gives you a quick walk-through of main AllenNLP concepts and features. We'll build a complete, working NLP model (text classifier) along the way.
* Part 2 provides in-depth tutorials on individual abstractions and features of AllenNLP.
* Part 3 introduces common NLP tasks and how to build models for specific tasks using AllenNLP.

This course contains plenty of code snippets and "hands-on" exercises that you can modify and run (powered by [Binder](https://mybinder.org/)). There is also [a companion repo](https://github.com/allenai/allennlp-course-examples) that contains full code needed to train and run the models in your environment. 

The entire course is written for AllenNLP version 1.0. AllenNLP 1.0+ is required to run the code snippets and exercises.

</exercise>

<exercise id="3" title="Target audience">

As a reader of this course, you need to be familiar with the Python programming language and PyTorch, the deep learning framework which AllenNLP is based on. You may also want to have a good understanding of modern deep NLP models and techniques and machine learning topics. However, you don't need any prior knowledge of AllenNLP—the whole point of this course is to provide onboarding of AllenNLP!

</exercise>

<exercise id="4" title="Table of contents">

- [Course overview](/overview)
  - What is AllenNLP
  - About this course
  - Target audience
  - Table of contents

## Part 1

- [Introduction](/introduction)
  - What is text classification
  - Defining input and output
* [Your first model](/your-first-model)
  - Reading data
  - Making a DatasetReader
  - Building your model
  - Implementing the model — the constructor
  - Implementing the model — the forward method
  - Writing a config file
* [Training and prediction](/training-and-prediction)
  - Training the model
  - Evaluating the model
  - Making predictions for unlabeled inputs
* [Next steps](/next-steps)
  - Switching to pre-trained contextualizers
  - More AllenNLP commands
  - Running a demo
  - Using GPUs

## Part 2

* Reading textual data
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
