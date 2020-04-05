# Splits text into words (instead of wordpieces or characters).
tokenizer = SpacyTokenizer()

# Represents each token with both an id from a vocabulary and a sequence of characters.
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab')
}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(['This', 'is', 'some', 'text', '.'],
                              namespace='token_vocab')
vocab.add_tokens_to_namespace(['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
                              namespace='character_vocab')

text = "This is some text."
tokens = tokenizer.tokenize(text)
print(tokens)

# The setup here is the same as what we saw above.
text_field = TextField(tokens, token_indexers)
text_field.index(vocab)
padding_lengths = text_field.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
# Note now that we have two entries in this output dictionary,
# one for each indexer that we specified.
print(tensor_dict)

# Split text into words with part-of-speech tags. This will result in the
# `tag_` variable being set on each `Token` object, which we will
# read in the indexer.
tokenizer = SpacyTokenizer(pos_tags=True)
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'],
                              namespace='pos_tag_vocab')

# Represents each token with (1) an id from a vocabulary, (2) a sequence of characters, and (3)
# part of speech tag ids.
token_indexers = {
    'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
    'token_characters': TokenCharactersIndexer(namespace='character_vocab'),
    'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab',
                                     feature_name='tag_'),
}

tokens = tokenizer.tokenize(text)
print(tokens)
print([token.tag_ for token in tokens])

text_field = TextField(tokens, token_indexers)
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print(tensor_dict)
