import torch
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PassThroughEncoder,
    LstmSeq2SeqEncoder,
)

batch_size = 8
sequence_length = 10
input_size = 5
hidden_size = 2

x = torch.rand(batch_size, sequence_length, input_size)
mask = torch.ones(batch_size, sequence_length)
print("shape of input:", x.shape)

encoder: Seq2SeqEncoder
encoder = PassThroughEncoder(input_dim=input_size)
y = encoder(x, mask=mask)

print("shape of output (PassThrough):", y.shape)

encoder = LstmSeq2SeqEncoder(input_size=input_size, hidden_size=hidden_size)
y = encoder(x, mask=mask)

print("shape of output (LSTM):", y.shape)
