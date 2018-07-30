""" Base Class and function for Decoders """

import torch
import torch.nn as nn
from .global_attention import GlobalAttention


class GRUDecoder(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.

    Args:
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout_prob (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, embeddings, hidden_size,
                 num_layers=1, dropout_prob=0.0, share_decoder_embeddings=True):
        super(GRUDecoder, self).__init__()

        # Basic attributes.
        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout_prob)

        # Build the RNN.
        self.gru = nn.GRU(input_size=self.embeddings.embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout_prob,
                          batch_first=True)

        # Set up the standard attention.
        self.attention_module = GlobalAttention(
            hidden_size
        )

        self.generator = nn.Sequential(
            # nn.Linear(hidden_size, self.embeddings.num_embeddings),
            nn.Linear(hidden_size, self.embeddings.embedding_dim),
            nn.Linear(self.embeddings.embedding_dim, self.embeddings.num_embeddings),
            # nn.LogSoftmax(dim=-1)
            )
        if share_decoder_embeddings:
            self.generator[1].weight = self.embeddings.weight

    def forward(self, targets, memory_bank, decoder_state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        assert isinstance(decoder_state, GRUDecoderState)
        assert targets.size(0) == memory_bank.size(0)  # equal batch_size

        # Run the forward pass of the RNN.
        embedded = self.embeddings(targets)

        # Run the forward pass of the RNN.
        rnn_output, hidden_state = self.gru(embedded, hx=decoder_state.hidden_state)

        # Calculate the attention.
        # decoder_outputs : (seq_len, batch_size, hidden_size)
        decoder_outputs, p_attn = self.attention_module(
            rnn_output.contiguous(),
            memory_bank,
            memory_lengths=memory_lengths
        )

        # decoder_outputs : (batch_size, seq_len, hidden_size)
        decoder_outputs = decoder_outputs.transpose(0, 1)  # fix the order of dimension
        attentions = {"std": p_attn}

        decoder_outputs = self.dropout_layer(decoder_outputs)

        # Update the state with the result.
        final_output = decoder_outputs[:, -1, :]
        decoder_state.hidden_state = hidden_state
        decoder_state.input_feed = final_output.unsqueeze(0)

        generated_outputs = self.generator(decoder_outputs)

        return generated_outputs, decoder_state, attentions

    def init_decoder_state(self, hidden_state):
        return GRUDecoderState(self.hidden_size, hidden_state)


class GRUDecoderState:
    """Interface for grouping together the current state of a recurrent
        decoder. In the simplest case just represents the hidden state of
        the model.  But can also be used for implementing various forms of
        input_feeding and non-recurrent models.

        Modules need to implement this to utilize beam search decoding.
        """
    """ Base class for RNN decoder state """

    def __init__(self, hidden_size, hidden_state):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            hidden_state: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        self.hidden_size = hidden_size
        self.hidden_state = hidden_state

        # Init the input feed.
        batch_size = self.hidden_state.size(1)
        self.input_feed = self.hidden_state.detach().new_zeros(batch_size, hidden_size).unsqueeze(0)

    @property
    def _all(self):
        return [self.hidden_state, self.input_feed]

    def update_state(self, hidden_state, input_feed):
        """ Update decoder state """
        self.hidden_state = hidden_state
        self.input_feed = input_feed

    # def repeat_beam_size_times(self, beam_size):
    #     """ Repeat beam_size times along batch dimension. """
    #     vars = [e.data.repeat(1, beam_size, 1)
    #             for e in self._all]
    #     self.hidden = tuple(vars[:-1])
    #     self.input_feed = vars[-1]
    #
    # def detach(self):
    #     """ Need to document this """
    #     self.hidden = tuple([_.detach() for _ in self.hidden])
    #     self.input_feed = self.input_feed.detach()
    #
    # def beam_update(self, idx, positions, beam_size):
    #     """ Need to document this """
    #     for e in self._all:
    #         sizes = e.size()
    #         br = sizes[1]
    #         if len(sizes) == 3:
    #             sent_states = e.view(sizes[0], beam_size, br // beam_size,
    #                                  sizes[2])[:, :, idx]
    #         else:
    #             sent_states = e.view(sizes[0], beam_size,
    #                                  br // beam_size,
    #                                  sizes[2],
    #                                  sizes[3])[:, :, idx]
    #
    #         sent_states.data.copy_(
    #             sent_states.data.index_select(1, positions))

if __name__ == '__main__':
    import numpy as np
    embeddings = nn.Embedding(1000, 128)
    decoder = GRUDecoder(embeddings, hidden_size=32, share_decoder_embeddings=False)
    targets = torch.tensor([[2, 3, 2, 2, 2, 3], [1,2, 2, 2, 0, 2], [1, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 2]])
    hidden = torch.tensor(np.random.random((1, 4, 32)), dtype=torch.float32)
    memory_bank = torch.tensor(np.random.random((4, 6, 32)), dtype=torch.float32)
    decoder_state = decoder.init_decoder_state(hidden)
    decoder_outputs, decoder_state, attentions = decoder(targets, memory_bank, decoder_state, memory_lengths=torch.tensor([6, 4, 1, 1]))
    print(decoder_outputs, decoder_state, attentions)
    print('decoder_outputs', decoder_outputs.shape)  # (seq_len, batch_size, hidden_size)
    print('decoder_state', decoder_state)
    print('attentions', attentions['std'].shape)  # (seq_len, batch_size, seq_len)

    print(decoder.embeddings.weight.shape)
    print(decoder.generator[1].weight.shape)