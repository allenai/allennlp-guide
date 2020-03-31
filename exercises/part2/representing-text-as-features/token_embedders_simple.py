# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer;
# Note that we added the batch dimension at the front.
token_tensor = {'tokens': {'tokens': torch.LongTensor([[1, 3, 2, 9, 4, 3]])}}

# You would typically get the number of embeddings here from the vocabulary;
# AllenNLP has a separate process for instantiating the Embedding object using
# the vocabulary that you don't need to worry about for now.
embedding = Embedding(num_embeddings=10, embedding_dim=3)
# This 'tokens' key matches the first 'tokens' key in the `token_tensor` above.
# We use these names to align the TokenIndexers used in the data code with
# the TokenEmbedders that do the work on the model side.
embedder = BasicTextFieldEmbedder(token_embedders={'tokens': embedding})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)

# This is what gets created by TextField.as_tensor with a TokenCharactersIndexer
# Note that we added the batch dimension at the front. Don't worry too much
# about the magic 'token_characters' key - that is hard-coded to be produced
# by the TokenCharactersIndexer, and accepted by TokenCharactersEncoder;
# you don't have to produce those yourself in normal settings, it's done for you.
token_tensor = {'tokens': {'token_characters': torch.tensor(
    [[[1, 3, 0], [4, 2, 3], [1, 9, 5], [6, 0, 0]]])}}

character_embedding = Embedding(num_embeddings=10, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=(3,))
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# Again here, the 'tokens' key is arbitrary. It just has to match whatever key
# you gave to the corresponding TokenIndexer in your data code, which ends up
# as the top-level key in the token_tensor dictionary.
embedder = BasicTextFieldEmbedder(token_embedders={'tokens': token_encoder})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
