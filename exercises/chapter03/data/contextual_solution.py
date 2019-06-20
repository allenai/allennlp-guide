from allennlp.data.fields import TextField
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data import Token, Vocabulary

# Splits text into words (instead of wordpieces or characters).  Underneath, in
# the model, BERT actually operates on a sequence of wordpieces, we just combine
# them back into words after getting BERT vectors for each wordpiece.
tokenizer = WordTokenizer()

# Represents each token with a sequences of wordpieces as determined by BERT.
# This actually results in a pretty complex data type, which you can see by
# running this.  It's complicated because we need to know how to combine the
# wordpieces back into words after running BERT.
# You could imagine just operating on wordpieces directly instead of combining
# things back into words. For that, you would really want a WordpieceTokenizer,
# which we don't have implemented (though you could likley grab one from other
# places).
token_indexer = PretrainedBertIndexer(pretrained_model='bert-base-cased')

# Both ELMo and BERT do their own thing with vocabularies, so we don't need to
# add anything, but we do need to construct the vocab object so we can use it
# below.
vocab = Vocabulary()

pre_text = "This is some text."
post_text = "With BERT, just include the tags inline."
pre_tokens = tokenizer.tokenize(pre_text)
post_tokens = tokenizer.tokenize(post_text)
# [CLS] gets automatically added at the beginning, and [SEP] at the end, so you
# only need to add a [SEP] in the middle if you have inputs you want to
# separate. Try removing this and seeing what happens, or adding more than one.
tokens = pre_tokens + [Token('[SEP]')] + post_tokens
print(tokens)

text_field = TextField(tokens, {'bert_tokens': token_indexer})
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print(tensor_dict)
