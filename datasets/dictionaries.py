from collections import Counter


class BaseDictionary:

    def __init__(self, data, vocabulary_size=None):
        self.vocabulary_size = vocabulary_size
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

        self.vocab_words, self.word2idx, self.idx2word = self._build_dictionary(data)

    def __getitem__(self, item):
        try:
            return self.word2idx[item]
        except KeyError:
            return self.word2idx[self.UNK_TOKEN]

    def _build_dictionary(self, data):

        counter = Counter([token for sentence in data for token in sentence])
        print("Total number of unique tokens:", len(counter))

        if self.vocabulary_size:
            counter = {word: freq for word, freq in
                       counter.most_common(self.vocabulary_size - 2)}  # - 2 for pad and unk
        else:
            counter = {word: freq for word, freq in counter}

        vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN] + list(sorted(counter.keys()))

        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words  # instead of {idx:word for idx, word in enumerate(vocab_words)}

        return vocab_words, word2idx, idx2word
