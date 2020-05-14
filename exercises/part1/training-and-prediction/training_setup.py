import tempfile
from typing import Dict, Iterable, List

import allennlp
import torch
from allennlp.common.params import Params
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training import Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.metrics import CategoricalAccuracy


class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                tokens = self.tokenizer.tokenize(text)
                if self.max_tokens:
                    tokens = tokens[:self.max_tokens]
                text_field = TextField(tokens, self.token_indexers)
                label_field = LabelField(sentiment)
                fields = {'text': text_field, 'label': label_field}
                yield Instance(fields)


class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {'loss': loss, 'probs': probs}


def build_dataset_reader() -> DatasetReader:
    return ClassificationTsvReader(max_tokens=64)

def read_data(
    reader: DatasetReader
) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    training_data = dataset_reader.read("quick_start/data/movie_review/train.tsv")
    validation_data = reader.read("quick_start/data/movie_review/dev.tsv")
    return training_data, validation_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)
