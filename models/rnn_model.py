from torch import nn

class RNNModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(RNNModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sources, inputs, source_lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):
                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """

        encoder_state, memory_bank = self.encoder(sources, source_lengths)
        decoder_state = self.decoder.init_decoder_state(encoder_state)
        decoder_outputs, decoder_state, attentions = self.decoder(inputs,
                                                                  memory_bank,
                                                                  decoder_state,
                                                                  memory_lengths=source_lengths)
        return decoder_outputs, decoder_state, attentions  # doesn't change the order
