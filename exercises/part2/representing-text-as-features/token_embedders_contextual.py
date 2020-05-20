# It's easiest to get ELMo input by just running the data code.  See the
# exercise above for an explanation of this code.
tokenizer = WhitespaceTokenizer()
token_indexer = ELMoTokenCharactersIndexer()
vocab = Vocabulary()
text = "This is some text."
tokens = tokenizer.tokenize(text)
print("ELMo tokens:", tokens)
text_field = TextField(tokens, {'elmo_tokens': token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("ELMo tensors:", token_tensor)

# We're using a tiny, toy version of ELMo to demonstrate this.
elmo_options_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/options.json'
elmo_weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/lm_weights.hdf5'
elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file,
                                   weight_file=elmo_weight_file)

embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("ELMo embedded tokens:", embedded_tokens)


# Again, it's easier to just run the data code to get the right output.

# We're using the smallest transformer model we can here, so that it runs on
# binder.
transformer_model = 'google/reformer-crime-and-punishment'
tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model)
token_indexer = PretrainedTransformerIndexer(model_name=transformer_model)
text = "Some text with an extraordinarily long identifier."
tokens = tokenizer.tokenize(text)
print("Transformer tokens:", tokens)
text_field = TextField(tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("Transformer tensors:", token_tensor)

embedding = PretrainedTransformerEmbedder(pretrained_model=transformer_model)

embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("Transformer embedded tokens:", embedded_tokens)
