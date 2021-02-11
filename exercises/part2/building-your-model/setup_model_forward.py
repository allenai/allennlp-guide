from typing import Dict

import torch
import numpy
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.models import Model
