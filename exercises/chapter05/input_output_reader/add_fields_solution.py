from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

# Data will be formatted as:
# [title][tab][text][tab][stars][tab][aspect][tab][sentiment]

@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                title, text, stars, aspect, sentiment = line.strip().split('\t')
                title_field = TextField(self.tokenizer.tokenize(title),
                                        self.token_indexers)
                text_field = TextField(self.tokenizer.tokenize(text),
                                       self.token_indexers)
                stars_field = LabelField(stars)
                aspect_field = LabelField(aspect)
                sentiment_field = LabelField(sentiment)
                fields = {'title': title_field,
                          'text': text_field,
                          'stars': stars_field,
                          'aspect': aspect_field,
                          'sentiment': sentiment_field}
                yield Instance(fields)

reader = ClassificationTsvReader()
print(reader.read('exercises/chapter05/input_output_reader/example_data.tsv'))
