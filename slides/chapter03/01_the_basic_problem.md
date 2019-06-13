---
type: slides
---

# How do you go from language to features?

---

# Language → Features

IMAGE of some text getting converted to numbers somehow

Notes: The basic problem in using machine learning on language data is converting symbolic text
into numerical features that can be used by machine learning algorithms.  In this chapter we will
focus on the predominant modern approach to solving this problem, using (perhaps contextualized)
word vectors; for a more in-depth discussion of and motivation for this approach, see [this
overview paper](https://arxiv.org/abs/1902.06006).

---

# Text → Tokens

IMAGE of text being broken up into tokens

Notes: The first part of representing a string of text as numerical features is splitting the text
up into individual `Tokens` that will each get their own representation.

---

# Tokens → Ids

IMAGE of tokens getting replaced by single ids

Notes: `Tokens` are then converted into numerical values using some pre-defined `Vocabulary`.

---

# Ids → Vectors

IMAGE of single ids getting replaced by word vectors

Notes: Finally, each individual id gets replaced by a vector representing that word in some
abstract space.  The idea here is that words that are "similar" to each other in some sense will be
close in the vector space, and so will be treated similarly by the model.

---

# Lots of options for these

IMAGE of using a char CNN

---

# Lots of options for these

IMAGE of using POS tag embeddings

---

# Lots of options for these

IMAGE of using both glove and char cnns

---

# Lots of options for these

IMAGE of using wordpieces and BERT

---

# Where do the pieces go?

There were three steps:

1. Text → Tokens
2. Tokens → Ids
3. Ids → Vectors

First two are data processing, last step is in the *model*

Notes: Of these three steps, only the last step has learnable paramaters.  There are certainly
decisions to be made in the first two steps that affect model performance, but they do not
typically have learnable parameters that you want to train with backprop on your final task.

This separation means that what originally looked like a simple problem (representing text as
features) actually needs coordination between two very different pieces of code: a `DatasetReader`,
that performs the first two steps, and a `Model`, that performs the last step.

---

# Core abstractions

1. Tokenizer (Text → Tokens)
2. TextField (Tokens → Ids)
3. TextFieldEmbedder (Ids → Vectors)

Notes: We want our code to not have to specify which of all of the many possible options we're
using to represent text as features; if we did that we would need to change our code to run simple
experiments.  Instead, we introduce high-level abstractions that encapsulate these operations and
write our code using the abstractions.  Then, when we run our code, we can construct the objects
with the particular versions of these abstractions that we want (in software engineering, this is
called _dependency injection_).

In the next set of slides and the following examples, we will examine the first two abstractions in
detail, showing how we go from text to something that can be input to a pytorch `Model`.
