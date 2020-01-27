from collections import Counter, defaultdict

from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary


# To create fields, simply pass the data to constructor.
# NOTE: we have a whole chapter on why TextFields are set up this way, and how they work.
tokens = [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')]
token_indexers = {'tokens': SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

label_field = LabelField('pos')

sequence_label_field = SequenceLabelField(['DET', 'ADJ', 'NOUN', 'ADV', 'PUNKT'], text_field)

# You can simply use print() fields to see their content
print(text_field)

print(label_field)

print(sequence_label_field)

# Fields know how to create empty fields of the same type
print(text_field.empty_field())

print(label_field.empty_field())

print(sequence_label_field.empty_field())

# You can count vocabulary items in fields
counter = defaultdict(Counter)
text_field.count_vocab_items(counter)
print(counter)

label_field.count_vocab_items(counter)
print(counter)

sequence_label_field.count_vocab_items(counter)
print(counter)

# Create Vocabulary for indexing fields
vocab = Vocabulary(counter)

# Fields know how to turn themselves into tensors
text_field.index(vocab)
# NOTE: in practice, we will batch together instances and use the maximum padding lengths,
# instead of getting them from a single instance
padding_lengths = text_field.get_padding_lengths()
print(text_field.as_tensor(padding_lengths))

label_field.index(vocab)
print(label_field.as_tensor({}))

sequence_label_field.index(vocab)
padding_lengths = sequence_label_field.get_padding_lengths()
print(sequence_label_field.as_tensor(padding_lengths))

# Fields know how to batch tensors
tensor1 = label_field.as_tensor({})

label_field2 = LabelField('pos')
label_field2.index(vocab)
tensor2 = label_field2.as_tensor({})

batched_tensors = label_field.batch_tensors([tensor1, tensor2])
print(batched_tensors)
