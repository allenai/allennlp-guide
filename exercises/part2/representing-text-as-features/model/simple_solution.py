from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
import torch

# This is what gets created by TextField.as_tensor with a
# TokenCharactersIndexer; see the exercises above.
token_tensor = {'tokens': torch.Tensor([[1, 3, 0], [4, 2, 3], [1, 9, 5], [6, 0, 0]])}

# You would typically get the number of embeddings here from the vocabulary;
# AllenNLP has a separate process for instantiating the Embedding object using
# the vocabulary that you don't need to worry about for now.
character_embedding = Embedding(num_embeddings=10, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=[3])
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

embedder = BasicTextFieldEmbedder(token_embedders={'tokens': token_encoder})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
