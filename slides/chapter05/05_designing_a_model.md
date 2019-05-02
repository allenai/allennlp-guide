---
type: slides
---

# Designing a Model

---

# What we need to do

Some figure showing `[batch of instances] --> [model] --> [loss]`

Notes: Now that we have a `DatasetReader` that produces `Instances` for us, the next thing we need
is a `Model` that will take a batch of `Instances`, predict the outputs from the inputs, and
compute a loss that we can use as a training signal.

After you've written these two pieces, AllenNLP takes care of the rest, connecting your input files
to the dataset reader, intelligently batching together your instances and feeding them to the
model, and optimizing the model's parameters by computing gradients using backprop on the loss.

---

# Our Instance format

Remember that our `Instances` have this input/output spec:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

Also, remember that we *used these names* (`text` and `label`) for the fields in the
`DatasetReader`. We need to use the same names in our model.

---

# What should our model do?

Figure showing `text --> embedding --> seq2vec encoding --> label`, as blobs

Notes: Conceptually, a generic model for classifying text looks something like this: you get some
features corresponding to each word in your input, then you combine those word-level features into
a document-level feature vector, then you classify that document-level feature vector into one of
your labels.

In AllenNLP, we make each of these conceptual steps into a generic abstraction that you can
use in your code, so that you can have a very flexible model that can use different concrete
components for each step, just by changing a configuration file.

---

# What should our model do?

Figure showing `text --> embedding --> seq2vec encoding --> label`, with `text` shown as a batch of
token ids

Notes: The first step is changing the strings in the input text into token ids, which is handled by
the `DatasetReader` (that's done by the `SingleIdTokenIndexer` that we used previously).

---

# What should our model do?

Figure showing `text --> embedding --> seq2vec encoding --> label`, with `text` and `embedding`
shown as tensors

Notes: The first thing our `Model` does is apply an `Embedding` function that converts each token
id that we got as input into a vector.  This gives us a vector for each input token, so we have a
large tensor here.

---

# What should our model do?

Figure showing `text --> embedding --> seq2vec encoding --> label`, with `text`, `embedding`, and
`seq2vec encoding` shown as tensors

Notes: Next we apply some function that takes the sequence of vectors for each input token and
squashing it into a single vector.  This is typically an LSTM or a convolutional encoder, with some
kind of pooling on top.

---

# What should our model do?

Figure showing `text --> embedding --> seq2vec encoding --> label`, with all of them shown as
tensors

Notes: Finally, we take that single feature vector (for each `Instance` in the batch), and classify
it as a label, which will give us a categorical probability distribution over our label space.

---

# AllenNLP Model basics

[some visualization here]

Notes: Now that we now what our model is going to do, we need to implement it.  First, we'll say a
few words about how `Models` work in AllenNLP.

An AllenNLP `Model` is just a pytorch `Module` with a particular output format (and some
other nice things we add that you don't need to worry about now).  Pytorch `Modules` implement a
`forward` method, which transform input tensors to output tensors using some parameters, providing
an easy way to train those parameters.  AllenNLP `Models` implement `forward`, and require the
output to be a _dictionary_ mapping strings to tensors, with a `loss` key provided during training.

Our training loop takes a batch of `Instances`, passes it through `Model.forward()`, grabs the
`loss` key from the resulting dictionary, and uses backprop to compute gradients and update the
model's parameters.
