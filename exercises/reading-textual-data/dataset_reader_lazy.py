from itertools import islice
from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer


@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        # To support laziness, constructor needs to take the lazy flag and pass it to the superclass
        super().__init__(lazy)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, tokens: List[Token], label: str = None) -> Instance:
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        # Also, make sure your _read() method yields instances (instead of returning a list)
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)
                yield self.text_to_instance(tokens, sentiment)


reader = ClassificationTsvReader(lazy=True)
dataset = reader.read('quick_start/data/movie_review/train.tsv')

# Returned dataset is a wrapped by a `LazyInstances` object
print('type of dataset: ', type(dataset))

# Preview the first 5 instances. This is when the _read() method is called for the first time
print('first 5 instances: ', list(islice(dataset, 5)))
