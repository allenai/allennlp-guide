from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

# It's easiest to get ELMo input by just running the data code.  See the
# exercise above for an explanation of this code.
tokenizer = WordTokenizer()
token_indexer = ELMoTokenCharactersIndexer()
vocab = Vocabulary()
text = "This is some text."
tokens = tokenizer.tokenize(text)
print(tokens)
text_field = TextField(tokens, {'elmo_tokens': token_indexer})
text_field.index(vocab)
padding_lengths = text_field.get_padding_lengths()
token_tensor = text_field.as_tensor(padding_lengths)
print(tensor_dict)

# We're using a tiny, toy version of ELMo to demonstrate this.
elmo_options_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/options.json'
elmo_weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/lm_weights.hdf5'
elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file,
                                   weight_file=elmo_weight_file)

embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
