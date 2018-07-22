from collections import Counter
from os.path import dirname, join, exists, abspath
from os import mkdir
import pickle

BASE_DIR = dirname(abspath(__file__))


class BaseDictionary:

    def __init__(self, name):

        try:
            saved_parameters = self.load(name)
        except FileNotFoundError:
            raise NotImplementedError("Prepare dictionary first")

        self.vocabulary_size = saved_parameters['vocabulary_size']
        self.vocab_words = saved_parameters['vocab_words']
        self.word2idx = saved_parameters['word2idx']
        self.idx2word = saved_parameters['idx2word']

        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

    def __getitem__(self, item):
        try:
            return self.word2idx[item]
        except KeyError:
            return self.word2idx[self.UNK_TOKEN]

    def prepare_dictionary(self, dataset):

        counter = Counter([token for sentence in dataset for token in sentence])
        print("Total number of unique tokens:", len(counter))

        if self.vocabulary_size:
            counter = {word: freq for word, freq in
                       counter.most_common(self.vocabulary_size - 2)}  # - 2 for pad and unk
        else:
            counter = {word: freq for word, freq in counter}

        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN] + list(sorted(counter.keys()))

        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words  # instead of {idx:word for idx, word in enumerate(vocab_words)}

        self.vocab_words, self.word2idx, self.idx2word = vocab_words, word2idx, idx2word

    def save(self, name):

        parameters = {'vocabulary_size': self.vocabulary_size,
                      'vocab_words': self.vocab_words,
                      'word2idx': self.word2idx,
                      'idx2word': self.idx2word}

        parameters_dir = join(BASE_DIR, 'parameters', 'base_dictionary')
        if not exists(parameters_dir):
            mkdir(parameters_dir)
        parameters_filepath = join(parameters_dir, name)

        with open(parameters_filepath, 'wb') as file:
            pickle.dump(parameters, file)

    @staticmethod
    def load(name):
        parameters_dir = join(BASE_DIR, 'parameters', 'base_dictionary')
        parameters_filepath = join(parameters_dir, name)

        with open(parameters_filepath, 'rb') as file:
            parameters = pickle.load(file)

        return parameters

    @staticmethod
    def exists_saved_parameters(name):
        parameters_dir = join(BASE_DIR, 'parameters', 'base_dictionary')
        parameters_filepath = join(parameters_dir, name)
        return exists(parameters_filepath)