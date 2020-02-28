import torch
from allennlp.modules.seq2seq_encoders.pass_through_encoder import PassThroughEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder

batch_size = 8
sequence_length = 10
input_size = 5
hidden_size = 2

x = torch.rand(batch_size, sequence_length, input_size)
mask = torch.ones(batch_size, sequence_length)
print('shape of input:', x.shape)

encoder = PassThroughEncoder(input_dim=input_size)
y = encoder(x, mask=mask)

print('shape of output (PassThrough):', y.shape)

encoder = LstmSeq2SeqEncoder(input_size=input_size, hidden_size=hidden_size)
y = encoder(x, mask=mask)

print('shape of output (LSTM):', y.shape)
