# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer;
# Note that we added the batch dimension at the front.  You choose the 'indexer1'
# name when you configure your data processing code.
token_tensor = {"indexer1": {"tokens": torch.LongTensor([[1, 3, 2, 9, 4, 3]])}}

# You would typically get the number of embeddings here from the vocabulary;
# if you use `allennlp train`, there is a separate process for instantiating the
# Embedding object using the vocabulary that you don't need to worry about for
# now.
embedding = Embedding(num_embeddings=10, embedding_dim=3)

# This 'indexer1' key must match the 'indexer1' key in the `token_tensor` above.
# We use these names to align the TokenIndexers used in the data code with the
# TokenEmbedders that do the work on the model side.
embedder = BasicTextFieldEmbedder(token_embedders={"indexer1": embedding})

embedded_tokens = embedder(token_tensor)
print("Using the TextFieldEmbedder:", embedded_tokens)

# As we've said a few times, what's going on inside is that we match keys between
# the token tensor and the token embedders, then pass the inner dictionary to the
# token embedder.  The above lines perform the following logic:
embedded_tokens = embedding(**token_tensor["indexer1"])
print("Using the Embedding directly:", embedded_tokens)

# This is what gets created by TextField.as_tensor with a TokenCharactersIndexer
# Note that we added the batch dimension at the front. Don't worry too much
# about the magic 'token_characters' key - that is hard-coded to be produced
# by the TokenCharactersIndexer, and accepted by TokenCharactersEncoder;
# you don't have to produce those yourself in normal settings, it's done for you.
token_tensor = {
    "indexer2": {
        "token_characters": torch.LongTensor(
            [[[1, 3, 0], [4, 2, 3], [1, 9, 5], [6, 0, 0]]]
        )
    }
}

character_embedding = Embedding(num_embeddings=10, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=(3,))
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# Again here, the 'indexer2' key is arbitrary. It just has to match whatever key
# you gave to the corresponding TokenIndexer in your data code, which ends up
# as the top-level key in the token_tensor dictionary.
embedder = BasicTextFieldEmbedder(token_embedders={"indexer2": token_encoder})

embedded_tokens = embedder(token_tensor)
print("With a character CNN:", embedded_tokens)
