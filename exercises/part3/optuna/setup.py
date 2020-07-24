import random

from allennlp.data import Vocabulary, allennlp_collate
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import GradientDescentTrainer
import numpy
import optuna
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from optuna.integration import AllenNLPPruningCallback
from optuna import Trial
