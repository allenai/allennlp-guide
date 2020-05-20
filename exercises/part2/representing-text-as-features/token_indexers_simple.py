# Splits text into words (instead of wordpieces or characters).
tokenizer = WhitespaceTokenizer()

# Represents each token with a single ID from a vocabulary.
token_indexer = SingleIdTokenIndexer(namespace='token_vocab')

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ['This', 'is', 'some', 'text', '.'],
    namespace='token_vocab')
vocab.add_tokens_to_namespace(
    ['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
    namespace='character_vocab')

text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("Word tokens:", tokens)

text_field = TextField(tokens, {'tokens': token_indexer})

# In order to convert the token strings into integer ids, we need to tell the
# TextField what Vocabulary to use.
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
# This output is pretty nested and might look complex.  The reason it is so
# nested is that we need to (1) align each indexer with a corresponding
# embedder in the model, and (2) pass a dictionary of arguments to the
# embedder by name.  This will be more clear when we get to the embedder.
print("With single id indexer:", tensor_dict)

# Represents each token with a sequence of characters.
token_indexer = TokenCharactersIndexer(namespace='character_vocab')

text_field = TextField(tokens, {'token_characters': token_indexer})
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
# Notice here that we've now got an extra dimension on the tensor - there's
# a sequence of characters for every token.
print("With token characters indexer:", tensor_dict)

# Splits text into characters (instead of words or wordpieces).
tokenizer = CharacterTokenizer()

tokens = tokenizer.tokenize(text)
print("Character tokens:", tokens)

# Represents each token (which is a character) as a single id from a vocabulary.
token_indexer = SingleIdTokenIndexer(namespace='character_vocab')

text_field = TextField(tokens, {'token_characters': token_indexer})
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("With single id indexer:", tensor_dict)
