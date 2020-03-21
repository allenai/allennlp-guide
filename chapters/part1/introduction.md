---
title: 'Introduction'
description:
  "This chapter will give an introduction to the task we'll be using throughout Part 2 (text classification) and how to use AllenNLP to solve it"
type: chapter
---

<textblock>

By this point, you should be familiar with what AllenNLP is and what it's useful for (see [Part 1](/overview) for review). In this section of the course, we'll give a quick start on one the most basic things you can do with AllenNLP: text classification.

</textblock>

<exercise id="1" title="What is text classification">

Text classification is one of the simplest NLP tasks where the model, given some input text, predicts a label for the text. See the figure below for an illustration.

<img src="/part1/introduction/text-classification.svg" alt="Text classification" />

There are a variety of applications of text classification, such as spam filtering, sentiment analysis, and topic detection, as shown in the table below.

| Application        | Description                        | Input                      | Output                              |
|--------------------|------------------------------------|----------------------------|-------------------------------------|
| Spam filtering     | Detect and filter spam emails etc. | Email                      | Spam / Not spam                     |
| Sentiment analysis | Detect the polarity of text        | Text (tweet, review, etc.) | Positive / Negative                 |
| Topic detection    | Detect the topic of text           | Text (news article, etc.)  | Business / Technology / Sports etc. |

</exercise>

<exercise id="2" title="Defining input and output">

The first step for building an NLP model is to define its input and output. In AllenNLP, each training example is represented by an [Instance](/reading-data) object. An Instance consists of one or more [Field](/reading-data)s, which contain data such as text and label. `Field`s will get converted to tensors and fed to your model. [Part 3](/reading-data) will provide more details on using `Instance`s and `Field`s to represent textual data.

For text classification, the input and the output are very simple. The model takes a `TextField` that represents the input text and predicts its label, which is represented by a `LabelField`:

```python
# Input
text: TextField

# Output
label: LabelField
```

In the next chapter, we'll cover how to write a `DatasetReader` to convert a dataset to a collection of `Instance`s and how to build your model.

</exercise>
