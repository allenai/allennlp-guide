---
title: 'Chapter 5: Text Classification'
description:
  "This chapter describes text classification and walks through an example of how to do it with
  AllenNLP.  As it's the first chapter talking about doing a practical task with AllenNLP, it goes
  into more detail about some of the basic components in AllenNLP, and how NLP works in general."
prev: /chapter04
next: /chapter06
type: chapter
id: 5
---

<textblock>

This is presented piece by piece, explaining things as we go.  To just skip to complete code for a
fully-featured classifier, see [this dataset
reader](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/text_classification_json.py)
and [this
model](https://github.com/allenai/allennlp/blob/master/allennlp/models/basic_classifier.py) in the
AllenNLP repository.

</textblock>



<exercise id="1" title="What is Text Classification?" type="slides">

<slides source="chapter05/01_what_is_text_classification" />

</exercise>




<exercise id="2" title="Varying the input/output spec">

Our running example will have only a single input and a single output, but there are lots of small
variations we could do on this theme.  Let's do some exercises to help you nail down how the
input/output spec translates into `Fields` in AllenNLP.

Let's say you're trying to detect sentiment in product reviews, which have a title in addition to a
review body.  How would you modify our spec to add the `title`?

<codeblock id="chapter05/input_output/add_title" executable="false">
The `title` field is also text.
</codeblock>

Now, in addition to the title, say we also have the number of stars the review got.  How might you
add that field as another input?

<codeblock id="chapter05/input_output/add_stars" executable="false">
The `stars` field is a label, not text.
</codeblock>

You might have thought giving the number of stars is basically the same as the sentiment; to make
it clear that it's different, let's add an "aspect" that the sentiment is about (say, the review
was positive overall, but negative about shipping).  We could decide to treat this as a model input
or as a model output.  How would we add it in either case?

<codeblock id="chapter05/input_output/add_aspect" executable="false">
The `aspect` field should probably be from a fixed set of categories, not free text.
</codeblock>

Really, though, there are probably several different aspects that we want to get sentiment about.
This is now getting somewhat complicated and outside the scope of "text classification", but let's
end with an exercise on what the input/output spec would be if we wanted to find all of the aspects
that were discussed in the review (from a fixed set of possible aspects) and the sentiment
associated with each of them.

<codeblock id="chapter05/input_output/add_list" executable="false">
There are lots of ways to do this, which all have different modeling implications. We're just
giving one possible option here.
</codeblock>

</exercise>



<exercise id="3" title="Reading Data" type="slides">

<slides source="chapter05/03_reading_data" />

</exercise>




<exercise id="4" title="Varying the input/output spec - modifying the DatasetReader">

Let's take one of our simple variations from the previous exercise and modify the `DatasetReader`
to match the new input/output spec.  How about the case where we have a title, text, number of
stars, and an aspect as input, and we're trying to predict sentiment about that aspect.  We give
some lines from an example input file at the top of the exercise.

<codeblock id="chapter05/input_output_reader/add_fields">
Look back at the previous exercise to see the kind of Fields that you should use.
</codeblock>

</exercise>



<exercise id="5" title="Designing a model" type="slides">

<slides source="chapter05/05_designing_a_model" />

</exercise>



<exercise id="6" title="Implementing the model - the constructor" type="slides">

<slides source="chapter05/06_model_constructor" />

</exercise>



<exercise id="7" title="Implementing the model - the forward method" type="slides">

<slides source="chapter05/07_model_forward" />

</exercise>



<exercise id="8" title="Putting it together">

In this exercise we'll put together a simple example of reading in data and feeding it to the
model.  We don't try to do any training, but this should let you play around with how the
`DatasetReader` and `Model` fit together.

<codeblock id="chapter05/putting_them_together/code">
Try adding a field to the `Instances` created by the `DatasetReader` and seeing how you have to
make corresponding changes to the `Model`.
</codeblock>

</exercise>



<exercise id="9" title="Using config files">

Here we'll give a simple example of creating the `DatasetReader` and `Model` from configuration
files, so you get some idea of how it works.  This is similar to the last exercise in that we'll
still try to put some data through the model to make sure everything works, but instead of manually
constructing objects, we'll use AllenNLP's `FromParams` magic.

<codeblock id="chapter05/putting_them_together/config">
Try changing the configuration parameters and see how the dataset reader and model change.  In
particular, see if you can add a character-level CNN to the `TextField` parameters.  You'll need to
add parameters both for the `DatasetReader` (inside a `token_indexers` block) and for the
`Model` (inside the `embedder` block).
</codeblock>

As you can see from these two exercises, if you're writing your own training script, there's not a
lot of difference in lines of code between the two.  But being able to instantiate everything from
config files means that we can write a general training loop for you in a way that you _never_ have
to write the instantiation code at all - you just give a configuration file, and we take care of
the rest.  It also makes it a lot easier to keep track of experiments that you run, because the
configuration for each experiment is already in a separate file, instead of buried inside of a
script somewhere.

</exercise>
