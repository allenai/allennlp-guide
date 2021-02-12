import json
import os
import tempfile
from copy import deepcopy
from typing import Dict, Iterable, List

import torch
from allennlp.common import JsonDict
from allennlp.common.params import Params
from allennlp.data import (
    Field,
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.models import Model
from allennlp.models.archival import archive_model, load_archive
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.predictors import Predictor
from allennlp.training import Trainer
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides


@DatasetReader.register("classification-tsv")
class ClassificationTsvReader(DatasetReader):
    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
    ):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields: Dict[str, Field] = {"text": text_field}
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                text, sentiment = line.strip().split("\t")
                yield self.text_to_instance(text, sentiment)


@Model.register("simple_classifier")
class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
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
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


def run_config(config):
    params = Params(json.loads(config))
    params_copy = params.duplicate()

    if "dataset_reader" in params:
        reader = DatasetReader.from_params(params.pop("dataset_reader"))
    else:
        raise RuntimeError("`dataset_reader` section is required")

    loader_params = params.pop("data_loader")
    train_data_loader = DataLoader.from_params(
        reader=reader,
        data_path=params.pop("train_data_path"),
        params=loader_params.duplicate(),
    )
    dev_data_loader = DataLoader.from_params(
        reader=reader,
        data_path=params.pop("validation_data_path"),
        params=loader_params,
    )

    print("Building the vocabulary...")
    vocab = Vocabulary.from_instances(train_data_loader.iter_instances())

    if "model" not in params:
        # 'dataset' mode — just preview the (first 10) instances
        print("Showing the first 10 instances:")
        for inst in train_data_loader.iter_instances():
            print(inst)
            return None

    model = Model.from_params(vocab=vocab, params=params.pop("model"))

    train_data_loader.index_with(vocab)
    dev_data_loader.index_with(vocab)

    # set up a temporary, empty directory for serialization
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = Trainer.from_params(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=train_data_loader,
            validation_data_loader=dev_data_loader,
            params=params.pop("trainer"),
        )
        trainer.train()

    return {
        "params": params_copy,
        "dataset_reader": reader,
        "vocab": vocab,
        "model": model,
    }
