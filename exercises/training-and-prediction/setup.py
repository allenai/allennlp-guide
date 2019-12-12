import json
import tempfile
from typing import Dict
from typing import Iterable

import torch
from allennlp.common.params import Params
from allennlp.data import DatasetReader, Instance
from allennlp.data import Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.iterators import DataIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training import Trainer


@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                text_field = TextField(self.tokenizer.tokenize(text),
                                       self.token_indexers)
                sentiment_field = LabelField(sentiment)
                fields = {'text': text_field, 'label': sentiment_field}
                yield Instance(fields)


@Model.register('simple_classifier')
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


def run_config(config):
    params = Params(json.loads(config))

    if 'dataset_reader' in params:
        reader = DatasetReader.from_params(params.pop('dataset_reader'))
    else:
        raise RuntimeError('`dataset_reader` section is required')

    all_instances = []
    if 'train_data_path' in params:
        train_data = reader.read(params.pop('train_data_path'))
        all_instances.extend(train_data)
    else:
        raise RuntimeError('`train_data_path` section is required')

    validation_data = None
    if 'validation_data_path' in params:
        validation_data = reader.read(params.pop('validation_data_path'))
        all_instances.extend(validation_data)

    vocab = Vocabulary.from_instances(all_instances)

    if 'model' not in params:
        # 'dataset' mode — just preview the (first 10) instances
        for inst in all_instances[:10]:
            print(inst)
    else:
        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        if 'trainer' not in params:
            if 'iterator' not in params:
                # 'forward instance' mode — feed instances to the model without training
                for inst in train_data:
                    outputs = model.forward_on_instance(inst)
                    print(outputs)
            else:
                # 'forward batch' mode — feed batches to the model without training
                iterator = DataIterator.from_params(params.pop('iterator'))
                iterator.index_with(vocab)

                for batch in iterator(train_data, num_epochs=1):
                    outputs = model(**batch)
                    print(outputs)
        else:
            # 'train' mode
            if 'iterator' not in params:
                raise RuntimeError('iterator is required for training')

            if not validation_data:
                raise RuntimeError('validation data is required for training')

            iterator = DataIterator.from_params(params.pop('iterator'))
            iterator.index_with(vocab)

            # set up a temporary, empty directory for serialization
            with tempfile.TemporaryDirectory() as serialization_dir:
                trainer = Trainer.from_params(
                    model=model,
                    serialization_dir=serialization_dir,
                    iterator=iterator,
                    train_data=train_data,
                    validation_data=validation_data,
                    params=params.pop('trainer'))
                trainer.train()
