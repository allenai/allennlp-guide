from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder
import torch

# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer;
# see the exercises above.
token_tensor = {'tokens': torch.Tensor([1, 3, 2, 9, 4, 3])}

# You would typically get the number of embeddings here from the vocabulary;
# AllenNLP has a separate process for instantiating the Embedding object using
# the vocabulary that you don't need to worry about for now.
embedding = Embedding(num_embeddings=10, embedding_dim=3)
embedder = BasicTextFieldEmbedder(token_embedders={'tokens': embedding})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
