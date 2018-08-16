from collections import Counter
from os.path import dirname, join, exists, abspath
from os import mkdir
import pickle
from tqdm import tqdm

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'
BASE_DIR = dirname(abspath(__file__))


class BaseDictionary:

    def __init__(self):

        self.vocabulary_size = None
        self.vocab_words, self.word2idx, self.idx2word = None, None, None

    def __getitem__(self, item):
        try:
            return self.word2idx[item]
        except KeyError:
            return self.word2idx[UNK_TOKEN]

    def prepare_dictionary(self, dataset, vocabulary_size=None, min_count=None):

        counter = Counter()
        for sentence in tqdm(dataset):
            counter.update(sentence)
        print("Total number of unique tokens:", len(counter))

        if vocabulary_size:
            counter = {word: freq for word, freq in
                       counter.most_common(vocabulary_size - 2)}  # - 2 for pad and unk
        if min_count is not None:
            counter = {word: freq for word, freq in counter.items()
                       if freq >= min_count}

        vocab_words = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN] + list(sorted(counter.keys()))

        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words  # instead of {idx:word for idx, word in enumerate(vocab_words)}

        self.vocabulary_size = len(vocab_words)
        self.vocab_words, self.word2idx, self.idx2word = vocab_words, word2idx, idx2word

    def save(self, save_name):

        parameters = {'vocabulary_size': self.vocabulary_size,
                      'vocab_words': self.vocab_words,
                      'word2idx': self.word2idx,
                      'idx2word': self.idx2word}

        parameters_dir = join(BASE_DIR, 'parameters', 'base_dictionary')
        if not exists(parameters_dir):
            mkdir(parameters_dir)
        parameters_filepath = join(parameters_dir, save_name)

        with open(parameters_filepath, 'wb') as file:
            pickle.dump(parameters, file)

    @classmethod
    def load(cls, save_name=None, parameters_filepath=None):

        if save_name is not None:
            parameters_dir = join(BASE_DIR, 'parameters', 'base_dictionary')
            parameters_filepath = join(parameters_dir, save_name)

        with open(parameters_filepath, 'rb') as file:
            saved_parameters = pickle.load(file)

        instance = cls()
        instance.vocabulary_size = saved_parameters['vocabulary_size']
        instance.vocab_words = saved_parameters['vocab_words']
        instance.word2idx = saved_parameters['word2idx']
        instance.idx2word = saved_parameters['idx2word']

        return instance

    @staticmethod
    def exists_saved_parameters(save_name):
        parameters_dir = join(BASE_DIR, 'parameters', 'base_dictionary')
        parameters_filepath = join(parameters_dir, save_name)
        return exists(parameters_filepath)

    def index_sentence(self, sentence):
        sentence = [START_TOKEN] + sentence + [END_TOKEN]
        return [self[word] for word in sentence]

    def index_chapter(self, chapter):
        chapter_indexed = []
        for sentence in chapter:
            sentence_indexed = self.index_sentence(sentence)
            chapter_indexed.append(sentence_indexed)
        return chapter_indexed
