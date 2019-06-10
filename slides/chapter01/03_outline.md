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

# [Chapter 2: AllenNLP's Main Abstractions](/chapter02)

- Basic components of the training loop (`Fields`, `Instances`, `DatasetReaders`, `Models`,
  `Trainers`, ...)

- How exactly is text input represented (the `TextField`, `TokenIndexers`, and `TokenEmbedders`):

- Abstractions for building models (these are all pytorch `Modules`; `Seq2VecEncoder`,
  `Seq2SeqEncoder`, `Attention`, `MatrixAttention`, `SpanEncoder`, ...)

Notes: This chapter describes the main abstractions we built into AllenNLP.  If you want to write
code using AllenNLP, it's really important to understand these abstractions and what they are used
for.
---

# Chapter 3: TextFields: A deep dive into the most important abstraction for NLP

- The main problem when applying machine learning to text input is deciding how to represent
  language as features in a model.

- There are a lot of ways to do this: one-hot ids, character sequences, pre-trained
  contextualizers, etc. `TextFields` are how AllenNLP abstracts away all of these options.

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

```json
{
  "dataset_reader": {
    "type": "snli"
  },
  "train_data_path": "/data/snli.json",
  "trainer": {
    "num_epochs": 50,
    ....
  }
  ...
}
```

Notes: AllenNLP provides a powerful and flexible mechanism for running experiments that are
entirely driven by `jsonnet` configuration files.  This chapter describes how that works and how to
use it.

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

# Chapter 14: Using pretrained contextualizers like ELMo and BERT

(I debated putting this before the task-specific stuff, because it's something people probably want
to see right away.  But I think you won't really understand this well without understanding how at
least one task works, so it really belongs after those chapters, even if someone who knows what
they're doing just skips to this.)

Notes: The best NLP systems today use large pre-trained contextualizers as their base inputs.  In
this chapter we give a brief introduction to pre-trained contextualizers and talk about how to use
them in AllenNLP.

---

# Chapter 15: Building a demo

(Include image of a demo here)

Notes: This chapter describes how AllenNLP is set up to making serving demos of your models easy,
and gives examples of how to do this, both for simple text-in / text-out demos, and for fancy
React-based visualizations.

---

# Chapter 16: Miscellaneous topics / FAQs

- How do I use a GPU instead of a CPU?  What about multiple GPUs?  How does multi-GPU support work?

Notes: A collection of random how-to topics and frequently asked questions that didn't fit in any other
chapter.
