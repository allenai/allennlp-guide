---
type: slides
---

# Designing a Model

---

# Designing a model

<img src="/your-first-model/designing-a-model.svg" alt="Designing a model" />

The next thing we need is a `Model` that will take a batch of `Instances`, predict the outputs from the inputs, and compute a loss.

After you've written `DatasetReader` and `Model`, AllenNLP takes care of the rest: connecting your input files to the dataset reader, intelligently batching together your instances and feeding them to the model, and optimizing the model's parameters by using backprop on the loss.

Notes:

---

# Our Instance format

Remember that our `Instances` have this input/output spec:

```python
# Inputs
text: TextField

# Outputs
label: LabelField
```

Also, remember that we *used these names* (`text` and `label`) for the fields in the `DatasetReader`. We need to use the same names in our model.

---

# What should our model do?

<img src="/your-first-model/designing-a-model-1.svg" alt="Designing a model 1" />

Conceptually, a generic model for classifying text does the following:

* Get some features corresponding to each word in your input
* Combine those word-level features into a document-level feature vector
* Classify that document-level feature vector into one of your labels.

In AllenNLP, we make each of these conceptual steps into a generic abstraction that you can use in your code, so that you can have a very flexible model that can use different concrete components for each step, just by changing a configuration file.

---

# Representing text with token IDs

<img src="/your-first-model/designing-a-model-2.svg" alt="Designing a model 2" />

The first step is changing the strings in the input text into token ids, which is handled by the `DatasetReader` (that's done by the `SingleIdTokenIndexer` that we used previously).

---

# Embedding tokens

<img src="/your-first-model/designing-a-model-3.svg" alt="Designing a model 3" />

The first thing our `Model` does is apply an `Embedding` function that converts each token ID that we got as input into a vector.  This gives us a vector for each input token, so we have a large tensor here.

---

# Apply Seq2Vec encoder

<img src="/your-first-model/designing-a-model-4.svg" alt="Designing a model 4" />

Next we apply some function that takes the sequence of vectors for each input token and
squashing it into a single vector.  This is typically an LSTM or a convolutional encoder, with some kind of pooling on top.

---

# Computing distribution over labels

<img src="/your-first-model/designing-a-model-5.svg" alt="Designing a model 5" />

Finally, we take that single feature vector (for each `Instance` in the batch), and classify it as a label, which will give us a categorical probability distribution over our label space.

---

# AlleNLP Model

<img src="/your-first-model/allennlp-model.svg" alt="AllenNLP model" />

* AllenNLP `Model` is just a PyTorch `Module`
* It implements a `forward()` method, and requires the output to be a _dictionary_
* Its output contains a `loss` key during training, which is used to optimize the model

Our training loop takes a batch of `Instances`, passes it through `Model.forward()`, grabs the `loss` key from the resulting dictionary, and uses backprop to compute gradients and update the model's parameters.
