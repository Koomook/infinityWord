from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
import math
from os.path import dirname, join, exists, abspath
import os
import pickle

BASE_DIR = dirname(abspath((__file__)))


class SoyTokenizer:

    def __init__(self, params_filepath=None):
        if params_filepath is None:
            params_filepath = join(BASE_DIR, 'parameters', 'soypreprocessor', 'scores.pkl')

        with open(params_filepath, 'rb') as file:
            self.scores = pickle.load(file)

        self.tokenizer = LTokenizer(self.scores)

    def tokenize(self, sentence):
        if sentence.startswith('⎡') or sentence.startswith('⎜'):
            sentence = sentence[0] + ' ' + sentence[1:]
        return self.tokenizer.tokenize(sentence)

    @staticmethod
    def prepare_preprocessor(dataset, **args):

        params_dir = join(BASE_DIR, 'parameters', 'soypreprocessor')
        if not exists(params_dir):
            os.makedirs(params_dir)

        word_extractor = WordExtractor(**args)
        word_extractor.train(dataset)
        words = word_extractor.extract()

        scores = {word: score.cohesion_forward * math.exp(score.right_branching_entropy) for word, score in words.items()}

        with open(join(params_dir, 'scores.pkl'), 'wb') as file:
            pickle.dump(scores, file)

    def tokenize_chapter(self, chapter):
        chapter_tokenized = []
        for sentence in chapter:
            sentence_tokenized = self.tokenize(sentence)
            chapter_tokenized.append(sentence_tokenized)
        return chapter_tokenized


"""
from preprocessors.soy_preprocessor import SoyPreprocessor

from datasets.base_datasets import SentencesTokenizedDataset
dataset = SentencesTokenizedDataset('train')
SoyPreprocessor.prepare_preprocessor([' '.join(l) for l in dataset])

soy_preprocessor = SoyPreprocessor()
soy_preprocessor.tokenize('안녕하세요')

from soynlp.noun import LRNounExtractor_v2
noun_extractor = LRNounExtractor_v2(verbose=True)
nouns = noun_extractor.train_extract([' '.join(soy_preprocessor.tokenize(' '.join(l))) for l in dataset])

"""