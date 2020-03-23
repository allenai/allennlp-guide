from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, PosTagIndexer
from allennlp.data.tokenizers import WordTokenizer, CharacterTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data import Vocabulary

# Splits text into words (instead of wordpieces or characters), and do part of speech tagging with
# spacy while we're at it.
tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))

# Represents each token with (1) an id from a vocabulary, (2) a sequence of characters, and (3)
# part of speech tag ids.
token_indexers = {'tokens': SingleIdTokenIndexer(namespace='token_vocab'),
                  'token_characters': TokenCharactersIndexer(namespace='character_vocab'),
                  'pos_tags': PosTagIndexer(namespace='pos_tag_vocab')}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(['This', 'is', 'some', 'text', '.'],
                              namespace='token_vocab')
vocab.add_tokens_to_namespace(['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
                              namespace='character_vocab')
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'],
                              namespace='pos_tag_vocab')

text = "This is some text."
tokens = tokenizer.tokenize(text)
print(tokens)
print([token.tag_ for token in tokens])

text_field = TextField(tokens, token_indexers)

# In order to convert the token strings into integer ids, we need to tell the
# TextField what Vocabulary to use.
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print(tensor_dict)
