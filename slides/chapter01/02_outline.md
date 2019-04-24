---
type: slides
---

# Course Outline

---

# Course written mid 2019, for AllenNLP version 1.0.

Notes: AllenNLP is a living software library, so things will get added or changed over time.
However, the main components of the library are stable, and the tasks described are timeless, so
the contents of this course should be applicable for all future versions.

---

# Chapter 2: AllenNLP's Main Abstractions

Basic components of the training loop:

- Fields
- Instances
- DatasetReaders
- Models
- Trainers

How exactly is text input represented:

- TextField
- TokenIndexer
- TokenEmbedder

Abstractions for building models:

- Seq2VecEncoder
- Seq2SeqEncoder
- SpanEncoder
- Attention
- MatrixAttention

---

# Chapter 3: AllenNLP's built-in commands

```bash
allennlp train CONFIG_FILE -s OUTPUT_DIR
allennlp evaluate ...
allennlp predict ...
allennlp dry-run ...
allennlp fine-tune ...
...
```


Notes: If you want to train a model, evaluate a model, make predictions using a model, or a number of
other common tasks, we have pre-built scripts that you can invoke from the command line.  This
chapter will go over how those work, how to use them, and how to write your own if you want.

---

# Chapter 4: Using AllenNLP's configuration files

---

# Chapters 5--12: A look at building models for specific tasks with AllenNLP

5. Text classification
6. Sentence pair classification
7. Sequence labeling
8. (Span classification)?  (I'm thinking SRL / coref)
9. Syntactic parsing (either dependency or constituency or both)
10. Reading comprehension
11. Language modeling
12. Sequence to sequence models
13. Semantic parsing

Notes: These chapters will describe the problem set up, how to encode the data in that setup using
AllenNLP's abstractions, what a simple model looks like, and how to write code that model in
AllenNLP.  These chapters are all based on models that can be found in the library, most of the
time with demos that are available.

---

# Chapter X: Using pretrained contextualizers like ELMo and BERT

(I debated putting this before the task-specific stuff, because it's something people probably want
to see right away.  But I think you won't really understand this well without understanding how at
least one task works, so it really belongs after those chapters, even if someone who knows what
they're doing just skips to this.)

---

# Chapter X: Miscellaneous topics

A collection of random how-to topics that didn't fit in any other chapter.
