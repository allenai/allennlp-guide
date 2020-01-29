import os
from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer


@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 cache_directory: str = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        # To enable caching, pass cache_directory to DatasetReader's constructor
        super().__init__(lazy, cache_directory)
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
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)
                yield self.text_to_instance(tokens, sentiment)


# Instantiate a dataset reader with a cache directory
reader = ClassificationTsvReader(cache_directory='dataset_cache')
dataset = reader.read('quick_start/data/movie_review/train.tsv')

# Check if the dataset is cached
print(os.listdir('dataset_cache'))

# The next time you invoke read(), instances are read from the cache
dataset = reader.read('quick_start/data/movie_review/train.tsv')
