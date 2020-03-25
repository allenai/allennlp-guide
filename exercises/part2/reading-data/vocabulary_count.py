from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

# Create fields and instances
token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens')}
text_field_pos = TextField(
    [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')],
    token_indexers=token_indexers)
text_field_neg = TextField(
    [Token('Such'), Token('an'), Token('awful'), Token('movie'), Token('.')],
    token_indexers=token_indexers)

label_field_pos = LabelField('pos', label_namespace='labels')
label_field_neg = LabelField('neg', label_namespace='labels')

instance_pos = Instance({'tokens': text_field_pos, 'label': label_field_pos})
instance_neg = Instance({'tokens': text_field_neg, 'label': label_field_neg})


# Create a Vocabulary with min_count=2 for tokens, but not for labels
vocab = Vocabulary.from_instances(
    [instance_pos, instance_neg],
    min_count={'tokens': 2})

print('Created a Vocabulary:', vocab)


# Getting the entire mapping for "tokens." Notice only 'movie' is in the vocabulary
print('index-to-token for "tokens":', vocab.get_index_to_token_vocabulary(namespace='tokens'))

# Getting the entire mapping for "labels." All the tokens that appeared in the dataset are present
print('index-to-token for "labels":', vocab.get_index_to_token_vocabulary(namespace='labels'))
