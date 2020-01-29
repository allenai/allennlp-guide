from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer


# You can implement your own dataset reader by subclassing the DatasetReader class
# At the very least, you need to implement the _read() method, preferably text_to_instance() as well
@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
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
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)
                yield self.text_to_instance(tokens, sentiment)


# Instantiate and use the dataset reader to read a file containing the data
reader = ClassificationTsvReader()
dataset = reader.read('quick_start/data/movie_review/train.tsv')

# Returned dataset is a list of Instances by default
print('type of dataset: ', type(dataset))
print('type of its first element: ', type(dataset[0]))
print('size of dataset: ', len(dataset))
