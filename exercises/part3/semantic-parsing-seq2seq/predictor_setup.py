from typing import List
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp_models.generation import ComposedSeq2Seq  # Need this for loading model archive

from nla_semparse.nla_semparse.nla_metric import NlaMetric  # Need this for loading model archive


archive = load_archive("nla_semparse/trained_models/seq2seq_model.tar.gz")
predictor = Predictor.from_archive(archive, "seq2seq")

def translate_nla(source: str) -> str:
    prediction_data = predictor.predict_json({"source": source})
    return " ".join(prediction_data["predicted_tokens"])
