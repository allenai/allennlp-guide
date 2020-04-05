from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Vocabulary

# Splits text into words (instead of wordpieces or characters).
tokenizer = WordTokenizer()

# Represents each token with an array of characters in a way that ELMo expects.
token_indexer = ELMoTokenCharactersIndexer()

# Both ELMo and BERT do their own thing with vocabularies, so we don't need to
# add anything, but we do need to construct the vocab object so we can use it
# below.
vocab = Vocabulary()

text = "This is some text."
tokens = tokenizer.tokenize(text)
print(tokens)

text_field = TextField(tokens, {'elmo_tokens': token_indexer})
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print(tensor_dict)
