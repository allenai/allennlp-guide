from typing import List
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp_models.generation import ComposedSeq2Seq  # Need this for loading model archive

from nla_semparse.nla_semparse.nla_metric import NlaMetric  # Need this for loading model archive


ARCHIVE = load_archive("nla_semparse/trained_models/seq2seq_model.tar.gz")
PREDICTOR = Predictor.from_archive(ARCHIVE, "seq2seq")

def translate_nla(source: str) -> List[str]:
    prediction_data = PREDICTOR.predict_json({"source": source})
    return prediction_data["predicted_tokens"]
