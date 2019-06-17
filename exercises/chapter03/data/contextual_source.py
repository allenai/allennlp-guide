from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer

# Splits text into words (instead of wordpieces or characters).
tokenizer = WordTokenizer()

# Represents each token with an array of characters in a way that ELMo expects.
token_indexer = ELMoTokenCharactersIndexer()

# Both ELMo and BERT do their own thing with vocabularies, so nothing is needed here.
vocab = Vocabulary()

text = "This is some text."
tokens = tokenizer.tokenize(text)
print(tokens)

text_field = TextField(tokens, {'elmo_tokens': token_indexer})

# In order to convert the token strings into integer ids, we need to tell the
# TextField what Vocabulary to use.
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print(tensor_dict)
