"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from .modules.positionwise_linear import PositionwiseFeedForward
from .modules.multi_headed_attention import MultiHeadedAttention
from .modules.layer_norm import LayerNorm

MAX_SIZE = 5000
PAD_TOKEN_ID = 0


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn_type="scaled-dot"):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn_type = self_attn_type

        if self_attn_type == "scaled-dot":  # Always
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        # print('inputs', inputs.shape)
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        # print('all_input', all_input.shape)
        # print('dec_mask', dec_mask.shape)
        query, attn = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        # print('memory_bank', memory_bank.shape)
        # print('query_norm', query_norm.shape)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, all_input

    @staticmethod
    def _get_attn_subsequent_mask(size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, share_decoder_embeddings=True):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.num_layers = num_layers
        self.embeddings = embeddings

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self.layer_norm = LayerNorm(d_model)

        self.generator = nn.Sequential(
            # nn.Linear(hidden_size, self.embeddings.num_embeddings),
            # nn.Linear(d_model, self.embeddings.embedding_dim),
            nn.Linear(self.embeddings.embedding_dim, self.embeddings.num_embeddings),
            # nn.LogSoftmax(dim=-1)
        )
        if share_decoder_embeddings:
            self.generator[0].weight = self.embeddings.weight

    def forward(self, tgt, memory_bank, state, memory_lengths=None,
                step=None, cache=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[batch x tgt_len]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[batch x src_len x hidden]`.
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
        # print('memory_bank', memory_bank.shape)
        # print('state cache', state.cache)
        src = state.src
        src_words = src
        tgt_words = tgt
        # print('hihi', src_words.size())
        src_batch, src_len = src_words.size()

        tgt_batch, tgt_len = tgt_words.size()
        # print('tgt_batch, tgt_len', tgt_batch, tgt_len)

        # Initialize return variables.
        outputs = []
        attns = {"std": []}

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)  #
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb
        src_memory_bank = memory_bank

        padding_idx = PAD_TOKEN_ID  # self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            # print('state.cache', state.cache)
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)] if state.cache is not None else None,
                    step=step)
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output
        attn = attn

        attns["std"] = attn

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        generated_outputs = self.generator(outputs)

        return generated_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers,
                              self.self_attn_type)
        return state


class TransformerDecoderState:
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers, self_attn_type):
        self.cache = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            if self_attn_type == "scaled-dot":
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            elif self_attn_type == "average":
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth))
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 1)
        if self.cache is not None:
            _recursive_map(self.cache)

    def beam_update(self, positions):
        """ Need to document this """

        for e in self._all:
            if len(e.size()) == 4: # if previous_layer_inputs
                e.data.copy_(e.data.index_select(1, positions))
            else:
                e.data.copy_(e.data.index_select(0, positions))