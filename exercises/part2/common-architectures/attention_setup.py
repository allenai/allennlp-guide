import torch
from allennlp.modules.attention import (
    Attention,
    DotProductAttention,
    BilinearAttention,
    LinearAttention,
)
from allennlp.modules.matrix_attention import (
    MatrixAttention,
    DotProductMatrixAttention,
    BilinearMatrixAttention,
    LinearMatrixAttention,
)
from allennlp.nn import Activation
