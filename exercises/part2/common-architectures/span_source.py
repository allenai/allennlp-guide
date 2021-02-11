# Create an instance with multiple spans
tokens = [
    Token(token)
    for token in ["I", "shot", "an", "elephant", "in", "my", "pajamas", "."]
]
token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

spans = [(2, 3), (5, 6)]  # ('an', 'elephant') and ('my', 'pajamas)
span_fields = ListField([SpanField(start, end, text_field) for start, end in spans])

instance = Instance({"tokens": text_field, "spans": span_fields})

# Alternatively, you can also enumerate all spans
spans = enumerate_spans(tokens, max_span_width=3)
print("all spans up to length 3:")
print(spans)


def filter_function(span_tokens):
    return not any(t == Token(".") for t in span_tokens)


spans = enumerate_spans(tokens, max_span_width=3, filter_function=filter_function)
print("all spans up to length 3, excluding punctuation:")
print(spans)


# Index and convert to tensors
vocab = Vocabulary.from_instances([instance])
instance.index_fields(vocab)

tensors = Batch([instance]).as_tensor_dict()
tokens_tensor, spans_tensor = tensors["tokens"], tensors["spans"]

# Embed the input
embedding_dim = 8
token_embedder = Embedding(embedding_dim=embedding_dim, vocab=vocab)
embedder = BasicTextFieldEmbedder({"tokens": token_embedder})
embedded_tokens = embedder(tokens_tensor)
print("shape of embedded_tokens", embedded_tokens.shape)
print("shape of spans_tensor:", spans_tensor.shape)  # type: ignore

# Embed the spans using two different span extractors
# combination='x,y' is the default value, but we are making it explicit here
span_extractor = EndpointSpanExtractor(input_dim=embedding_dim, combination="x,y")
embedded_spans = span_extractor(embedded_tokens, spans_tensor)

print("shape of embedded spans (x,y):", embedded_spans.shape)

span_extractor = EndpointSpanExtractor(input_dim=embedding_dim, combination="x-y")
embedded_spans = span_extractor(embedded_tokens, spans_tensor)

print("shape of embedded spans (x-y):", embedded_spans.shape)
