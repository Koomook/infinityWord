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


class CandidatesTransformerGenerator:

    def __init__(self, model, preprocess, postprocess, checkpoint_filepath=None, max_length=30, beam_size=8):
        self.model = model
        self.preprocess = preprocess  # lambda source: [dictionary[word] for word in sum([sentence.split() for sentence in source], [])]
        self.postprocess = postprocess  # lambda h: ' '.join([self.dictionary.idx2word[token_id] for token_id in h])
        self.max_length = max_length
        self.beam_size = beam_size

        if checkpoint_filepath is not None:
            checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.attentions = None

    def generate_candidates(self, source, n_candidates=5):

        source_preprocessed = self.preprocess(source)
        source_tensor = torch.tensor([source_preprocessed])
        # print('source_tensor', source_tensor.shape)

        sources_mask = self.model.pad_masking(source_tensor, source_tensor.size(1))
        memory_mask = self.model.pad_masking(source_tensor, 1)
        memory = self.model.encoder(source_tensor, sources_mask)

        decoder_state = self.model.decoder.init_decoder_state()
        # print('decoder_state src', decoder_state.src.shape)
        # print('previous_input previous_input', decoder_state.previous_input)
        # print('previous_input previous_layer_inputs ', decoder_state.previous_layer_inputs)


        # Repeat beam_size times
        memory_beam = memory.detach().repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, hidden_size)

        beam = Beam(beam_size=self.beam_size, min_length=0, n_top=n_candidates, ranker=None)

        for _ in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            decoder_outputs, decoder_state = self.model.decoder(new_inputs, memory_beam,
                                                                            memory_mask,
                                                                            state=decoder_state)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            attention = self.model.decoder.decoder_layers[-1].memory_attention_layer.sublayer.attention
            beam.advance(decoder_outputs.squeeze(1), attention)

            beam_current_origin = beam.get_current_origin()  # (beam_size, )
            decoder_state.beam_update(beam_current_origin)

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=n_candidates)
        hypothesises, attentions = [], []
        for i, (times, k) in enumerate(ks[:n_candidates]):
            hypothesis, attention = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
            attentions.append(attention)

        self.attentions = attentions
        hs = [self.postprocess(h) for h in hypothesises]
        return hs


class CandidatesGenerator:

    def __init__(self, model, preprocess, postprocess, checkpoint_filepath, max_length=30, beam_size=8):
        self.model = model
        self.preprocess = preprocess  # lambda source: [dictionary[word] for word in sum([sentence.split() for sentence in source], [])]
        self.postprocess = postprocess  # lambda h: ' '.join([self.dictionary.idx2word[token_id] for token_id in h])
        self.max_length = max_length
        self.beam_size = beam_size

        checkpoint = torch.load(checkpoint_filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.attentions = None

    def generate_candidates(self, source, n_candidates=5):

        source_preprocessed = self.preprocess(source)
        source_tensor = torch.tensor([source_preprocessed])
        source_length = torch.tensor([len(source_preprocessed)])
        # print('source_tensor', source_tensor.shape)
        encoder_state, memory_bank = self.model.encoder(source_tensor, source_length)
        encoder_state = encoder_state.transpose(0, 1)
        # print('encoder_state: (num_layers, batch_size=1, hidden_size)', encoder_state.shape)
        # print('memory_bank: (batch_size, seq_len, hidden_size)', memory_bank.shape)
        # encoder_state: (num_layers, batch_size=1, hidden_size)
        # memory_bank: (batch_size, seq_len, hidden_size)

        decoder_state = self.model.decoder.init_decoder_state(source_tensor, memory_bank, encoder_state)
        # print('decoder_state src', decoder_state.src.shape)
        # print('previous_input previous_input', decoder_state.previous_input)
        # print('previous_input previous_layer_inputs ', decoder_state.previous_layer_inputs)

        # Repeat beam_size times
        memory_bank_beam = memory_bank.detach().repeat(self.beam_size, 1, 1)  # (beam_size, seq_len, hidden_size)
        memory_length_beam = source_length.repeat(self.beam_size)  # (beam_size, )
        decoder_state.repeat_beam_size_times(self.beam_size)

        beam = Beam(beam_size=self.beam_size, min_length=0, n_top=n_candidates, ranker=None)

        for _ in range(self.max_length):

            new_inputs = beam.get_current_state().unsqueeze(1)  # (beam_size, seq_len=1)
            decoder_outputs, decoder_state, attentions = self.model.decoder(new_inputs, memory_bank_beam,
                                                                            decoder_state,
                                                                            memory_length_beam)
            # decoder_outputs: (beam_size, target_seq_len=1, vocabulary_size)
            # attentions['std']: (target_seq_len=1, beam_size, source_seq_len)

            beam.advance(decoder_outputs.squeeze(1), attentions['std'])

            beam_current_origin = beam.get_current_origin()  # (beam_size, )
            decoder_state.beam_update(beam_current_origin)

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=n_candidates)
        hypothesises, attentions = [], []
        for i, (times, k) in enumerate(ks[:n_candidates]):
            hypothesis, attention = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
            attentions.append(attention)

        self.attentions = attentions
        hs = [self.postprocess(h) for h in hypothesises]
        return hs