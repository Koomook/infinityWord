
import torch
from torch import nn
from os.path import dirname, abspath, join
import sys

BASE_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(BASE_DIR)

from models.transformer_encoder import TransformerEncoder
from models.transformer_decoder import TransformerDecoder

embeddings = nn.Embedding(100, 512)
encoder = TransformerEncoder(num_layers=2, d_model=512, heads=8, d_ff=512, dropout=0.2, embeddings=embeddings)
example_source = torch.tensor([[54, 23, 10], [10, 99, 0], [1, 0, 0], [1, 0, 0]])
hidden_state, rnn_output = encoder(example_source, [3, 2, 1, 1 ])
print(hidden_state.shape, hidden_state.dtype)
print(rnn_output.shape, rnn_output.dtype)

import numpy as np
embeddings = nn.Embedding(1000, 512)
decoder = TransformerDecoder(num_layers=2, d_model=512, heads=8, d_ff=512, dropout=0.2, embeddings=embeddings)
targets = torch.tensor([[2, 3, 2, 2, 2, 3], [1,2, 2, 2, 0, 2], [1, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 2]])
hidden = torch.tensor(np.random.random((1, 4, 32)), dtype=torch.float32)
memory_bank = torch.tensor(np.random.random((4, 6, 512)), dtype=torch.float32)
decoder_state = decoder.init_decoder_state(targets, memory_bank, hidden_state)
decoder_outputs, decoder_state, attentions = decoder(targets, memory_bank, decoder_state, memory_lengths=torch.tensor([6, 4, 1, 1]))
print(decoder_outputs, decoder_state, attentions)
print('decoder_outputs', decoder_outputs.shape)  # (seq_len, batch_size, hidden_size)
print('decoder_state', decoder_state)
print('attentions', attentions['std'].shape)  # (seq_len, batch_size, seq_len)

print(decoder.embeddings.weight.shape)
print(decoder.generator[1].weight.shape)