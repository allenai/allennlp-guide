from allennlp.data.fields import TextField
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Token, Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedBertEmbedder

# It's easiest to get BERT input by just running the data code.  See the
# exercise above for an explanation of this code.
tokenizer = WordTokenizer()
token_indexer = PretrainedBertIndexer(pretrained_model='bert-base-cased')
vocab = Vocabulary()
pre_text = "This is some text."
post_text = "With BERT, just include the tags inline."
pre_tokens = tokenizer.tokenize(pre_text)
post_tokens = tokenizer.tokenize(post_text)
tokens = [Token('[CLS]')] + pre_tokens + [Token('[SEP]')] + post_tokens
print(tokens)
text_field = TextField(tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)
padding_lengths = text_field.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
print(tensor_dict)

# We don't have a tiny version of BERT, so this is using the full model and
# might be a bit slow.
bert_embedding = PretrainedBertEmbedder(pretrained_model='bert-base-uncased')

embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': bert_embedding})

embedded_tokens = embedder(token_tensor)
print(embedded_tokens)
