
# This pattern is typically used in cases where your input data is already
# tokenized, so we're showing that here.
text_tokens = ["This", "is", "some", "frandibulous", "text", "."]
tokens = [Token(x) for x in text_tokens]
print(tokens)

# We're using a very small transformer here so that it runs quickly in binder. You
# can change this to any transformer model name supported by Hugging Face.
transformer_model = 'google/reformer-crime-and-punishment'

# Represents the list of word tokens with a sequences of wordpieces as determined
# by the transformer's tokenizer.  This actually results in a pretty complex data
# type, which you can see by running this.  It's complicated because we need to
# know how to combine the wordpieces back into words after running the
# transformer.
indexer = PretrainedTransformerMismatchedIndexer(model_name=transformer_model)

text_field = TextField(tokens, {'transformer': indexer})
text_field.index(Vocabulary())
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())

# There are two key things to notice in this output.  First, there are two masks:
# `mask` is a word-level mask that gets used in the utility functions described in
# the last section of this chapter.  `wordpiece_mask` gets used by the `Embedder`
# itself.  Second, there is an `offsets` tensor that gives start and end wordpiece
# indices for the original tokens.  In the embedder, we grab these, average all of
# the wordpieces for each token, and return the result.
print("Indexed tensors:", token_tensor)

embedding = PretrainedTransformerMismatchedEmbedder(model_name=transformer_model)

embedder = BasicTextFieldEmbedder(token_embedders={'transformer': embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("Embedded tokens size:", embedded_tokens.size())
print("Embedded tokens:", embedded_tokens)
