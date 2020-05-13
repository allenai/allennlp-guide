from collections import Counter, defaultdict

from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
