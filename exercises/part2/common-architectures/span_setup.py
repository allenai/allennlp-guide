from typing import Dict

import torch

from allennlp.data import Batch, Instance, Token, Vocabulary
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.data.fields import TextField, ListField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
