import torch
from allennlp.modules.attention import (
    DotProductAttention,
    BilinearAttention,
    LinearAttention,
)
from allennlp.modules.matrix_attention import (
    DotProductMatrixAttention,
    BilinearMatrixAttention,
    LinearMatrixAttention,
)
from allennlp.nn import Activation
