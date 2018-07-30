import torch
from torch.nn.functional import softmax
import numpy as np
from .beam import Beam

START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'


class OneSentenceGenerator:

    def __init__(self, model, dictionary, checkpoint_filepath, max_length=30):
        self.model = model
        self.dictionary = dictionary
        self.max_length = max_length

        checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def generate_one(self):

        generated_sentence = []
        generated_token_ids = [self.dictionary[START_TOKEN]]

        for _ in range(self.max_length):
            inputs = torch.tensor(generated_token_ids).unsqueeze(0)
            outputs = self.model(inputs)
            next_token_probs = softmax(outputs[0, -1, :], dim=0)
            next_token_id = next_token_probs.argmax().item()
            next_token = self.dictionary.idx2word[next_token_id]
            if next_token == END_TOKEN:
                break
            generated_sentence.append(next_token)
            generated_token_ids.append(next_token_id)

        return ' '.join(generated_sentence)


class OneSentenceGeneratorWithBeam:

    def __init__(self, preprocess, encoder, decoder, dictionary, checkpoint_filepath, max_length=30, beam_size=8):
        self.preprocess = preprocess
        self.encoder = encoder
        self.decoder = decoder
        self.dictionary = dictionary
        self.max_length = max_length
        self.beam_size = beam_size

        checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def generate_one(self, source):

        beam = Beam(beam_size=self.beam_size, min_length=0, n_top=1, ranker=None)

        src_lengths = None
        encoder_states, memory_bank = self.model.encoder(source, src_lengths)
        decoder_states = self.model.decoder.init_decoder_state(
            source, memory_bank, encoder_states)
        memory_lengths = src_lengths.repeat(beam_size)

        for _ in range(self.max_length):
            if beam.done():
                break

            beam_current_state = beam.get_current_state()
            decoder_inputs = torch.tensor(beam_current_state)

            dec_out, decoder_states, attn = self.model.decoder(
                decoder_inputs, memory_bank, decoder_states, memory_lengths=memory_lengths,
                step=i)
            outputs = self.model(inputs)
            beam.advance(outputs)

            if beam.done():
                break

        return beam.pick_best()