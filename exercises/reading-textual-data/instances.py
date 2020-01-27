from collections import Counter, defaultdict

from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

# Create Fields
tokens = [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')]
token_indexers = {'tokens': SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

label_field = LabelField('pos')

sequence_label_field = SequenceLabelField(['DET', 'ADJ', 'NOUN', 'ADV', 'PUNKT'], text_field)

# Create a Vocabulary
counter = defaultdict(Counter)
text_field.count_vocab_items(counter)
label_field.count_vocab_items(counter)
sequence_label_field.count_vocab_items(counter)

vocab = Vocabulary(counter)

# Create an Instance
fields = {
    'tokens': text_field,
    'label': label_field,
}
instance = Instance(fields)

# You can add fields later
instance.add_field('label_seq', sequence_label_field)

# You can simply use print() to see the instance's content
print(instance)

# Index all fields in the instance by calling index_fields()
instance.index_fields(vocab)

# Instances know how to turn themselves into a dict of tensors
print(instance.as_tensor_dict())
