from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
import torch

# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer, a
# TokenCharactersIndexer, and a PosTagIndexer; see the exercises above.
token_tensor = {'tokens': torch.Tensor([2, 4, 3, 5]),
                'token_characters': torch.Tensor([[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]]),
                'pos_tag_tokens': torch.tensor([2, 5, 3, 4])}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(['This', 'is', 'some', 'text', '.'],
                              namespace='token_vocab')
vocab.add_tokens_to_namespace(['T', 'h', 'i', 's', ' ', 'o', 'm', 'e', 't', 'x', '.'],
                              namespace='character_vocab')
vocab.add_tokens_to_namespace(['DT', 'VBZ', 'NN', '.'],
                              namespace='pos_tag_vocab')

# Notice below how the 'vocab_namespace' parameter matches the namespace used above; typically
# you'll pass that namespace to the `Indexer` class (or use its default), and pass the _same_ value
# to the embedding class, as shown below.

# This is for embedding each token.
embedding = Embedding.from_params(vocab, Params({'vocab_namespace': 'token_vocab',
                                                 'embedding_dim': 3}))

# This is for encoding the characters in each token.
character_embedding = Embedding.from_params(vocab, Params({'vocab_namespace': 'character_vocab',
                                                           'embedding_dim': 4}))
cnn_encoder = CnnEncoder(embedding_dim=4, num_filters=5, ngram_filter_sizes=[3])
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# This is for embedding the part of speech tag of each token.
pos_tag_embedding = Embedding.from_params(vocab, Params({'vocab_namespace': 'pos_tag_vocab',
                                                         'embedding_dim': 6}))

# Notice how these keys match the keys in the token_tensor dictionary above; these are the keys
# that you give to your TokenIndexers when constructing your TextFields in the DatasetReader.
# Matching these pieces is essential to having everything fit together correctly in AllenNLP.
embedder = BasicTextFieldEmbedder(token_embedders={'tokens': embedding,
                                                   'token_characters': token_encoder,
                                                   'pos_tag_tokens': pos_tag_embedding})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
