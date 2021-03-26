from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.models.archival import load_archive
from allennlp_models.classification.dataset_readers import StanfordSentimentTreeBankDatasetReader

import warnings; warnings.filterwarnings("ignore")  # just to suppress CUDA warning for better showing
