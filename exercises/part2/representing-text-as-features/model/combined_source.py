from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
import torch

# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer and a
# TokenCharactersIndexer; see the exercises above.
token_tensor = {'tokens': torch.Tensor([2, 4, 3, 5]),
                'token_characters': torch.Tensor([[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]])}

# You would typically get the number of embeddings here from the vocabulary;
# Click on "show solution" below to see a simplified version of how this happens.

# This is for embedding each token.
embedding = Embedding(num_embeddings=6, embedding_dim=3)

# This is for encoding the characters in each token.
character_embedding = Embedding(num_embeddings=6, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=[3])
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

embedder = BasicTextFieldEmbedder(token_embedders={'tokens': embedding,
                                                   'token_characters': token_encoder})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
