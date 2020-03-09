from allennlp.data.fields import TextField, ListField, SpanField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.batch import Batch


# Create an instance with multiple spans
tokens = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas', '.']
tokens = [Token(token) for token in tokens]
token_indexers = {'tokens': SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

spans = [(2, 3), (5, 6)]    # ('an', 'elephant') and ('my', 'pajamas)
span_fields = ListField([SpanField(start, end, text_field) for start, end in spans])

instance = Instance({
    'tokens': text_field,
    'spans': span_fields
})

# Index and convert to tensors
vocab = Vocabulary.from_instances([instance])
instance.index_fields(vocab)

tensors = Batch([instance]).as_tensor_dict()
tokens_tensor, spans_tensor = tensors['tokens'], tensors['spans']

# Embed the input
embedding_dim = 8
token_embedder = Embedding(embedding_dim=embedding_dim, vocab=vocab)
embedder = BasicTextFieldEmbedder({'tokens': token_embedder})
embedded_tokens = embedder(tokens_tensor)
print('shape of embedded_tokens', embedded_tokens.shape)
print('shape of spans_tensor:', spans_tensor.shape)

# Embed the spans using two different span extractors
# combination='x,y' is the default value, but we are making it explicit here
span_extractor = EndpointSpanExtractor(input_dim=embedding_dim, combination='x,y')
embedded_spans = span_extractor(embedded_tokens, spans_tensor)

print('shape of embedded spans (x,y):', embedded_spans.shape)

span_extractor = EndpointSpanExtractor(input_dim=embedding_dim, combination='x-y')
embedded_spans = span_extractor(embedded_tokens, spans_tensor)

print('shape of embedded spans (x-y):', embedded_spans.shape)
