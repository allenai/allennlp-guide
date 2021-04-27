from typing import Dict

from allennlp.data import DatasetReader, Instance, TokenIndexer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary


class MyDatasetReader(DatasetReader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._token_indexers: Dict[str, TokenIndexer] = {
            "tokens": SingleIdTokenIndexer()
        }

    def _read(self, file_path):
        for tokens, label in zip(
            [["a", "b", "c", "d"], ["e"], ["f", "g", "h"], ["i", "j"]],
            ["a", "b", "c", "d"],
        ):
            yield Instance(
                {
                    "tokens": TextField(
                        [Token(t) for t in tokens], self._token_indexers
                    ),
                    "label": LabelField(label),
                }
            )
