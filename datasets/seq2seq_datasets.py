from .base_datasets import ChaptersTokenizedDataset
from pymongo import MongoClient
from tqdm import tqdm
import json
from os.path import dirname, abspath, join

START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'
BASE_DIR = dirname(dirname(abspath(__file__)))
MONGODB_CONFIG = json.load(open(join(BASE_DIR, 'datasets', 'mongodb_config.json')))

CLIENT = MongoClient(host=MONGODB_CONFIG['host'],
                     username=MONGODB_CONFIG['username'],
                     password=MONGODB_CONFIG['password'],
                     authSource=MONGODB_CONFIG['authSource'],
                     authMechanism=MONGODB_CONFIG['authMechanism'])
DB = CLIENT.get_database(MONGODB_CONFIG['database'])


class OneSeq2SeqDataset:

    def __init__(self, phase):

        assert phase in ('train', 'val', 'test')
        self.collection = DB.get_collection('novels_sources_targets')
        self.cursor = self.collection.find({'phase': phase})

    def __getitem__(self, index):
        sentence_document = self.cursor[index]
        source = sentence_document['text']['source']
        target = sentence_document['text']['target']
        return source, target

    def __len__(self):
        return self.cursor.count()

    def __iter__(self):
        self.cursor.rewind()
        for sentence_document in self.cursor:
            source = sentence_document['text']['source']
            target = sentence_document['text']['target']
            return source, target

    @staticmethod
    def prepare_dataset():
        source_length = 1

        target_collection = DB.get_collection('novels_sources_targets')
        target_collection.remove({})
        for phase in ['train', 'val', 'test']:
            source_collection = DB.get_collection('novels_' + phase + '_chapters')

            for chapter in tqdm(source_collection.find()):
                tokenized_chapter = ChaptersTokenizedDataset.tokenize_chapter(chapter['text'])
                if len(tokenized_chapter) <= source_length:
                    continue

                source = []
                for sentence_index, sentence in enumerate(tokenized_chapter):
                    if len(source) == source_length:
                        sentence_document = {
                            'phase': phase,
                            'novel_id': chapter['novel_id'],
                            'chapter_index': chapter['chapter_index'],
                            'target_sentence_index': sentence_index,
                            'text': {
                                'source': source,
                                'target': sentence
                            }
                        }
                        target_collection.insert_one(sentence_document)
                        source = source[1:]
                        source.append(sentence)
                    else:
                        source.append(sentence)


class Seq2SeqIndexedDataset:

    def __init__(self, phase):

        self.collection = DB.get_collection('novels_sources_targets')
        self.cursor = self.collection.find({'phase':phase})

    def __getitem__(self, item):
        document = self.cursor[item]
        source_indexed = document['indexed']['source']
        target_indexed = document['indexed']['target']
        return source_indexed, target_indexed

    def __len__(self):
        return self.cursor.count()

    @staticmethod
    def prepare_dataset(dictionary):

        for phase in ['train', 'val', 'test']:
            source_collection = DB.get_collection('novels_sources_targets')
            data = source_collection.find({'phase': phase})
            for sentence_document in tqdm(data):
                source_sentences = sentence_document['text']['source']
                target_sentence = sentence_document['text']['target']

                inputs_indexed = [Seq2SeqIndexedDataset.index_sentence(source_sentence, dictionary) for source_sentence in source_sentences]
                targets_indexed = Seq2SeqIndexedDataset.index_sentence(target_sentence, dictionary)
                sentence_document_update = {
                    "$set": {
                    'indexed': {'source': inputs_indexed,
                                'target': targets_indexed}}
                }
                source_collection.update_one({'_id': sentence_document['_id']}, sentence_document_update)

    @staticmethod
    def index_sentence(sentence, dictionary):
        sentence = [START_TOKEN] + sentence + [END_TOKEN]
        return [dictionary[word] for word in sentence]


if __name__ == '__main__':

    # OneSeq2SeqDataset.prepare_dataset()
    one_seq2seq_dataset = OneSeq2SeqDataset('train')
    print('one_seq2seq_dataset[0]', one_seq2seq_dataset[0])

    from os.path import dirname, abspath
    import sys

    BASE_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(BASE_DIR)

    from dictionaries import BaseDictionary

    dictionary = BaseDictionary.load('base_dictionary')

    Seq2SeqIndexedDataset.prepare_dataset(dictionary)
    seq2seq_indexed_dataset = Seq2SeqIndexedDataset('train')
    print('seq2seq_indexed_dataset[0]', seq2seq_indexed_dataset[0])
