---
type: slides
---

# What is Text Classification?

---

# What is Text Classification?

IMAGE of text to label, maybe with an example from twitter sentiment about AllenNLP

Notes: This is the simplest thing you can do with natural language processing - given some input
text, predict a label for the text.  This could be spam detection in an email client, topic
classification for a news site, sentiment detection on twitter, or a host of other possible
applications.

---

# Prerequisites

We assume you are familiar with the following:

- [Your First Model](chapter01#2), where we go over the basics of training an AllenNLP model
- We'll use some of the abstractions described in [Chapter 3](chapter03/), in particular the [data
  abstractions](#) (including [TextField](#) and [LabelField](#)) and the [model abstractions](#)
(including the [TextFieldEmbedder](#) and the [Seq2VecEncoder](#)).  We'll also briefly describe
these as we go, if you haven't seen them before.

---

# Input / Output Spec

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

Notes: For any task you want to do with AllenNLP, the first thing to do is be clear about what your
inputs and outputs are.  For text classification, our inputs and outputs are simple: a single piece
of text as input, and a single label as output.
