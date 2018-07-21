import torch
from torch import nn
import numpy as np


class LSTM(nn.Module):

    def __init__(self, vocabulary_size=10000):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=1024, num_layers=2,
                            batch_first=True, dropout=0.5, bidirectional=False)
        self.decoder = nn.Linear(in_features=1024, out_features=vocabulary_size)

    def forward(self, inputs):

        inputs_embedded = self.embedding(inputs)
        lstm_output, (h_n, c_n) = self.lstm(inputs_embedded)  # (batch_size, seq_len, hidden_size * num_directions)
        return self.decoder(lstm_output)


if __name__ == '__main__':
    model = LSTM(vocabulary_size=10000)
    print('model', model)
    example_inputs = np.random.randint(low=0, high=10000, size=(8, 10))
    print('example_inputs', example_inputs)
    example_tensor = torch.tensor(example_inputs)
    example_output = model(example_tensor)
    print('example_output', example_output)
