from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.attackers import InputReduction
from allennlp_models.classification.dataset_readers import (
    StanfordSentimentTreeBankDatasetReader,
)

import warnings

warnings.filterwarnings("ignore")  # just to suppress CUDA warning for better showing
