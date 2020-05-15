import torch
from allennlp.modules.seq2vec_encoders import CnnEncoder, LstmSeq2VecEncoder

batch_size = 8
sequence_length = 10
input_size = 5
hidden_size = 2

x = torch.rand(batch_size, sequence_length, input_size)
mask = torch.ones(batch_size, sequence_length)
print('shape of input:', x.shape)

encoder = LstmSeq2VecEncoder(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=1)
y = encoder(x, mask=mask)

print('shape of output (LSTM):', y.shape)

encoder = CnnEncoder(embedding_dim=input_size,
                     num_filters=1,
                     output_dim=hidden_size)
y = encoder(x, mask=mask)

print('shape of output (CNN):', y.shape)
