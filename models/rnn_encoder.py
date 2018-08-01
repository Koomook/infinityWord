import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class GRUEncoder(nn.Module):
    """ A GRU neural network encoder.

    Args:
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
    """

    def __init__(self, embeddings, hidden_size,
                 bidirectional=False, num_layers=1, dropout_prob=0.0):
        super(GRUEncoder, self).__init__()

        self.embeddings = embeddings
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=self.embeddings.embedding_dim,
                          hidden_size=self.hidden_size,
                          num_layers=num_layers,
                          dropout=dropout_prob,
                          bidirectional=bidirectional,
                          batch_first=True)

        self.bridge = nn.Linear(in_features=hidden_size * num_layers ,
                                out_features=hidden_size * num_layers,
                                bias=True)

    def forward(self, source, lengths):

        batch_size = source.size(0)

        emb = self.embeddings(source)
        packed_emb = pack(emb, lengths, batch_first=True)

        rnn_output, hidden_state = self.gru(packed_emb)
        # rnn_output : (batch, seq_len, hidden_size * num_directions)
        # hidden_state : (num_layers * num_directions, batch, hidden_size)

        rnn_output, lengths = unpack(rnn_output, batch_first=True)

        if self.bidirectional:
            hidden_state = hidden_state.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

            forward_state = hidden_state[:, 0, :, :]
            backward_state = hidden_state[:, 1, :, :]
            hidden_state = torch.cat([forward_state, backward_state], dim=2)

        hidden_state = self.bridge(hidden_state)

        return hidden_state, rnn_output


if __name__ == '__main__':
    embeddings = nn.Embedding(100, 28)
    encoder = GRUEncoder(embeddings, 32, bidirectional=True)
    example_source = torch.tensor([[54, 23, 10], [10, 99, 0], [1, 0, 0]])
    hidden_state, rnn_output = encoder(example_source, [3, 2, 1])
    print(hidden_state.shape, hidden_state.dtype)
    print(rnn_output.shape, rnn_output.dtype)
