# We're following the logic from the "Combining multiple TokenIndexers" example
# above.
tokenizer = SpacyTokenizer(pos_tags=True)

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ['This', 'is', 'some', 'text', '.'],
    namespace='token_vocab')
vocab.add_tokens_to_namespace(
    ['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
    namespace='character_vocab')
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'],
                              namespace='pos_tag_vocab')

text = "This is some text."
text2 = "This is some text with more tokens."
tokens = tokenizer.tokenize(text)
tokens2 = tokenizer.tokenize(text2)
print("Tokens:", tokens)
print("Tokens 2:", tokens2)


# Represents each token with (1) an id from a vocabulary, (2) a sequence of
# characters, and (3) part of speech tag ids.
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab'),
    'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab',
                                     feature_name='tag_'),
}

text_field = TextField(tokens, token_indexers)
text_field.index(vocab)
text_field2 = TextField(tokens2, token_indexers)
text_field2.index(vocab)

# We're using the longer padding lengths here; we'd typically be relying on our
# collate function to figure out what the longest values are to use.
padding_lengths = text_field2.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
tensor_dict2 = text_field2.as_tensor(padding_lengths)
print("Combined tensor dictionary:", tensor_dict)
print("Combined tensor dictionary 2:", tensor_dict2)

text_field_tensors = text_field.batch_tensors([tensor_dict, tensor_dict2])
print("Batched tensor dictionary:", text_field_tensors)

# We've seen plenty of examples of using a TextFieldEmbedder, so we'll just show
# the utility methods here.
mask = nn_util.get_text_field_mask(text_field_tensors)
print("Mask:", mask)
print("Mask size:", mask.size())
token_ids = nn_util.get_token_ids_from_text_field_tensors(text_field_tensors)
print("Token ids:", token_ids)

# We can also handle getting masks when you have lists of TextFields, but there's
# an important parameter that you need to pass, which we'll show here.  The
# difference in output that you see between here and above is just that there's an
# extra dimension in this output.  Where shapes used to be (batch_size=2, ...),
# now they are (batch_size=1, list_length=2, ...).
list_field = ListField([text_field, text_field2])
tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
text_field_tensors = list_field.batch_tensors([tensor_dict])
print("Batched tensors for ListField[TextField]:", text_field_tensors)

# The num_wrapping_dims argument tells get_text_field_mask how many nested lists
# there are around the TextField, which we need for our heuristics that guess
# which tensor to use when computing a mask.
mask = nn_util.get_text_field_mask(text_field_tensors, num_wrapping_dims=1)
print("Mask:", mask)
print("Mask:", mask.size())
