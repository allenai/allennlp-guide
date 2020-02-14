from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer()

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)


CONFIG = """
{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 8
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
"""

components = run_config(CONFIG)
dataset_reader, vocab, model = components['dataset_reader'], components['vocab'], components['model']

predictor = SentenceClassifierPredictor(model, dataset_reader)

output = predictor.predict('A good movie!')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])
output = predictor.predict('This was a monstrous waste of time.')
print([(vocab.get_token_from_index(label_id, 'labels'), prob)
       for label_id, prob in enumerate(output['probs'])])
