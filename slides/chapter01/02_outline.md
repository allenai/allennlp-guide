---
type: slides
---

# Course Outline

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

# Chapters 3--10: A look at building models for specific tasks with AllenNLP

3. Text classification
4. Sentence pair classification
5. Sequence labeling
6. (Span classification)?  (I'm thinking SRL / coref)
7. Syntactic parsing (either dependency or constituency or both)
8. Reading comprehension
9. Sequence to sequence models
10. Semantic parsing

Notes: These chapters will describe the problem set up, how to encode the data in that setup using
AllenNLP's abstractions, what a simple model looks like, and how to write code that model in
AllenNLP.  These chapters are all based on models that can be found in the library, most of the
time with demos that are available.

---

# Chapter 11: Using pretrained contextualizers like ELMo and BERT

---

# Chapter 12: Using AllenNLP's configuration files

---

# Chapter 13: Miscellaneous topics

A collection of random how-to topics that didn't fit in any other chapter.
