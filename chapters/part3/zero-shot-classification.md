---
title: 'Zero-shot classification'
description: 'In this chapter we build several models for performing zero-shot classification.'
type: chapter
---

<textblock>UNDER CONSTRUCTION</textblock>

<exercise id="1" title="What is zero-shot classification?">

In text classification, we want to assign some kind of label to a given piece of text.  Typically,
the label set is known, and the modeling problem is relatively straightforward.  It's simple enough
that we covered this problem in our [quick start](/introduction).  Text classification includes
things like topic classification of news articles, sentiment classification of reviews or tweets,
and a host of other related problem.

Sometimes, however, you want to assign a label to a piece of text without ever having seen training
data with that label.  We call this setting _zero-shot classification_, because you have zero
examples of the label in your training set.  You might encounter this setting if you want to
classify news articles by topic, but you don't have training data for all of the topics that you
want to classify into, or several other related scenarios.[^1]

[^1]: Fundamentally, zero-shot classification is a format that has arbitrary scope—you could pose
  almost any problem that you want as a zero-shot classification problem.  For example, I could have
  my label be "sarcastic documents using outdated metaphors"; if the "label" is allowed to be
  compositional language, things could get very complicated very fast. We'll stick to
  single-concept, non-compositional zero-shot labels in this chapter.

Zero-shot classification has been studied in NLP for a long time (at least [since
2008](https://www.aaai.org/Papers/AAAI/2008/AAAI08-132.pdf)).  We are going to use a particular
formulation and dataset from an ACL 2019 paper titled [Benchmarking Zero-shot Text Classification:
Datasets, Evaluation and Entailment Approach](https://www.aclweb.org/anthology/D19-1404/).  This
dataset has three categories of labels: topic ("health", "sports", "politics", etc.), emotion
("joy", "anger", "disgust", etc.), and situation ("medical assistance", "regime change", etc.).
We're just going to use the topic and emotion portions of the data, as the situation category is a
multi-label problem, and we're not going to worry about handling that complexity in this chapter.
Some examples from the dataset are below.


    Family & Relationships
    What makes friendship click ? How does the spark keep going ? good communication is what does it . Can you move beyond small talk and say what 's really on your mind . If you start doing this , my expereince is that potentially good friends will respond or shun you . Then you know who the really good friends are .

    Science & Mathematics
    Why does Zebras have stripes ? What is the purpose or those stripes ? Who do they serve the Zebras in the wild life ? this provides camouflage - predator vision is such that it is usually difficult for them to see complex patterns

    Education & Reference
    What did the itsy bitsy sipder climb up ? waterspout

    fear
    My car skidded on the wet street.

    fear
    When we stayed in Vienna with our class, my friend and I behaved incorrectly. Our teacher threatened us with exclusion from school.

    joy
    When I met my girlfriend - I had not counted on that.

    sadness
    In a social situation I became interested in a woman. We talked, we laughed, we enjoyed each other. She desappeared for a few minutes, and a little after appeared with an other man.

    shame
    In the traffic I insulted a man who crossed my way. Afterwards I paired with him, and felt shame because of mine lack of pacience and ridiculous attitude.

    guilt
    A great friend of mine travelled with the intention to change his life. He didn't succeed and returned depressed. I had not power to support his frustration and his behavioral change.

Our task will be to try to classify these documents into their respective labels without having seen
any examples of the label, or even in the entire label category.

</exercise>

<exercise id="2" title="Making a classifier zero-shot: removing the final softmax layer">

Having now understood the problem setting, we turn to the main question this chapter tries to
address: how do you take a text classification model and make it work for a zero-shot classification
problem?

A classification model is about as simple as you can get in NLP—you just embed and encode your text,
then use a simple linear layer to project a hidden dimension to a scalar value for each class, which
is converted to a probability distribution using a softmax operation.  We'll repeat a figure from
[the quick start chapter](http://localhost:8001/your-first-model#3) that shows what this model looks
like:

<img src="/part1/your-first-model/designing-a-model-5.svg" alt="Designing a model 5" />

The fundamental problem that we have to solve to make this into a zero-shot classification model is
the last arrow in that figure, the part of the model that takes us from a hidden vector representing
each document to a distribution over labels.  In our simple classifier that was a pytorch `Linear`
layer that knew about the number of labels in the data:

<pre data-line="11" class="language-python line-numbers"><code>
@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
</code></pre>

That `Linear` layer has a weight matrix with shape `(encoding_dim, num_labels)`.  You can interpret
this as a hidden vector representing each label, and the classifier just takes the hidden vector for
each document and does a dot product with each label's vector, giving a score for how similar they
are.

The trouble is that the vector for each label had to be _learned_.  Without any data to learn a
vector for the label, what's our basis for making a prediction?  We need some way to get a vector
for each label without using any training data.  If we get a good vector for the label that is
"compatible" in some sense with the document encodings that our model produces, then we can have
some hope of performing zero-shot classification for that label.

There are a number of different possible strategies to recover this vector for each label (or do
something similar with a different model structure).  In the rest of this chapter, we'll talk about
five of them, giving example model code and showing how well the alternatives work.[^2]  All of the
options we consider except the last one will make the assumption that for any given input we have a
list of _candidate_ labels (some of which may be entirely unseen) that we can classify into.

[^2]: We present experimental results here, but they shouldn't be interpreted to be particularly
  rigorous. We did no hyper-parameter tuning, and it's possible that for each case that we looked at
  there are model tweaks we could have made to improve performance. The main point here is to give
  you an idea of how these models work and how to write code for them.

</exercise>

<exercise id="3" title="Replacing the softmax layer with an embedding layer">
</exercise>

<exercise id="4" title="Encoding a label description for the final layer">

Take a longer textual description, encode it, use it as your set of labels.

</exercise>

<exercise id="5" title="Sentence pair classification">

Straightforward, but you have to run the model once per label

</exercise>

<exercise id="6" title="Question answering">

Can make this effectively multiple choice, so you only have to run the model once.  Harder as the
number of labels grows, though.

</exercise>

<exercise id="7" title="Generative modeling">

Main problem: huge vocabulary space.

</exercise>

<exercise id="8" title="Serving a demo of this model">

</exercise>
