import os, sys
import re
from gensim.models import Word2Vec, FastText
import gensim

sys.path.append(sys.path[0] + "/..")
from ..utils.mongo import connect_mongo


class FileIterator(object):
    """Create sentence generator which yield line by line.
    """
    def __init__(self, 
                 target_dirpath=None, 
                 min_count=None, 
                 duplicate=False, 
                 fundamental_dirpath='data/word2vec_train/mecab/fundamental', 
                 target_txt_path=None, 
                 use_fundamental=True):

        if target_txt_path:

            self.file_path = [target_txt_path]

        else:

            self.target_file_path_list = [os.path.join(target_dirpath, fname) for fname in os.listdir(target_dirpath) if re.search('txt', fname)]
            if duplicate:
                self.target_file_path_list = self.target_file_path_list*min_count

            if use_fundamental:
                self.fundamental_dirpath = [os.path.join(fundamental_dirpath, fname) for fname in os.listdir(fundamental_dirpath) if re.search('txt', fname)]
                self.file_path = self.target_file_path_list + self.fundamental_dirpath
            else:
                self.file_path = self.target_file_path_list

    def __iter__(self):

        def _read_dir():
            for fpath in self.file_path:
                read = open(fpath, encoding='utf-8')
                line = None
                while line != '':
                    line = read.readline()
                    yield line.split()
                print('{} done'.format(fpath))

        return _read_dir()

class SentenceIterator(object):
    """iterate sentences from mongodb"""
    def __init__(self, db, novel_collection, lyrics_collection, namu=None):
        self.db = db
        self.novel_collection = novel_collection
        self.lyrics_collection = lyrics_collection
        if namu:
            self.namu = True
            self.namu_iter = FileIterator(target_txt_path=namu)
        else:
            self.namu = False
    
    def get_classname(self, iterator):
        return str(iterator.__class__).split('.')[-1][:-2]

    def count_interval(self, interval=1000000):
        if self.count % interval == 0:
            print('count {}'.format(self.count))

    def get_iterators(self):
        iterators = [self.db.novel(self.novel_collection), 
                     self.db.lyrics(self.lyrics_collection)]
        if self.namu:
            iterators = iterators +[self.namu_iter]
        return iterators

    def test(self):
        def test_mongo(iterator, key1, key2, key3):
            for doc in iterator:
                data = doc[key1][key2][key3]
                for text in data:
                    for line in text:
                        words = line.split()
                        if len(words) <= 3:
                            continue
                        # todo: if 한자 : continue
                        print(['SOS/TK'] + words + ['EOS/TK'])
                        return True

        def test_txt(iterator):
            for words in iterator:
                print(['SOS/TK'] + words + ['EOS/TK'])
                return True

        iter_set = [test_mongo(self.db.novel(self.novel_collection), 'text', 'tagged', 'mecab'), 
                    test_mongo(self.db.lyrics(self.lyrics_collection), 'lyrics', 'processed', 'mecab')]
        if self.namu:
            iter_set.append(test_txt(self.namu_iter))

        if iter_set == [True]*len(iter_set):
            print('All test succeded')
            return None
        else:
            raise NotImplementedError

    def __iter__(self):
        """yield sentence"""
        self.count = 0
        iterators = self.get_iterators()
        for iterator in iterators:
            print(iterator)
            if self.get_classname(iterator) == 'FileIterator':
                for words in iterator:
                    yield ['SOS/TK'] + words + ['EOS/TK']
                    self.count += 1
                    self.count_interval()
            else:
                for doc in iterator:
                    if iterator.collection.name == 'books_converted':
                        data = doc['text']['tagged']['mecab']
                    elif iterator.collection.name == 'processed':
                        data = doc['lyrics']['processed']['mecab']
                    else:
                        raise NotImplementedError
                    for text in data:
                        for line in text:
                            words = line.split()
                            if len(words) <= 3:
                                continue
                            # todo: if 한자 : continue
                            yield ['SOS/TK'] + words + ['EOS/TK']
                            self.count += 1
                            self.count_interval()

class WordEmbedding(object):
    """Run gensim word2vec library for word embedding
    
    
    """
    def __init__(self, method):

        self.method = method
        if method == 'word2vec':
            self.embedding = gensim.models.word2vec.Word2Vec
        elif method == 'fasttext':
            self.embedding = FastText
    
    def run(self, iterator, window, size, min_count, n_iter=10, workers=32):
        # self.sentences = Sentence_gen(self.txt_dir_path, self.min_count, duplicate=self.duplicate, use_fundamental=self.use_fundamental)
        self.model = self.embedding(iterator, min_count=min_count, iter=n_iter, window=window, size=size, workers=workers)
        self.line_count = iterator.count
        print(self.line_count)

    def run_continued(self, iterator, epochs=10):
        self.model.build_vocab(iterator, update=True)
        self.model.train(iterator, self.model.corpus_count, epochs=epochs)
        print(iterator.count)

    def save(self, save_dir, save_parents_class=True):
        """save as KeyedVector format"""
        try:
            os.mkdir(save_dir)
        except OSError:
            pass
        file_name = 'window{}_size{}_min{}_{}'.format(self.model.window, self.model.vector_size, self.model.vocabulary.min_count, self.method)
        self.model.wv.save(os.path.join(save_dir, '{}_{}'.format('wv', file_name)))
        if save_parents_class:
            self.model.save(os.path.join(save_dir, '{}_{}'.format(self.method, file_name)))

class OnlineLearning(object):
    """online learning from pre trained word2vec model

    """
    def __init__(self, w2v_path, model_output_dir, model_name=None, txt_dir_path=None, 
                 target_txt_path=None, min_count_for_sentence=None, duplicate=False):
        try:
            os.mkdir(model_output_dir)
        except OSError:
            pass

        if txt_dir_path == None and target_txt_path == None:
            raise AssertionError

        self.model = Word2Vec.load(w2v_path)
        original_name = w2v_path.split('/')[-1]
        if model_name == None:
            self.model_name = 'onlinelearning_{}'.format(original_name)
        else:
            self.model_name = '{}_{}'.format(model_name, original_name)
        self.model_output_path = os.path.join(model_output_dir, self.model_name)
        
        self.sentences = Sentence_gen(target_dirpath=txt_dir_path, min_count=min_count_for_sentence, target_txt_path=target_txt_path, duplicate=duplicate)

    def run(self, epochs=10):
        self.model.build_vocab(self.sentences, update=True)
        self.model.train(self.sentences, total_examples=self.model.corpus_count, epochs=epochs)

    def save(self):
        self.model.save(self.model_output_path)
    
