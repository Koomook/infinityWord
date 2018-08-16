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
    def prepare_dataset(source_length=1):

        source_collection = DB.get_collection('novels_chapters')
        target_collection = DB.get_collection('novels_sources_targets')
        target_collection.drop()

        cursor = source_collection.find().batch_size(70)
        for chapter in tqdm(cursor):
            tokenized_chapter = chapter['text_tokenized']
            if len(tokenized_chapter) <= source_length:
                continue

            source = []
            for sentence_index, sentence in enumerate(tokenized_chapter):
                if len(source) == source_length:
                    sentence_document = {
                        'phase': chapter['phase'],
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

    def __init__(self, phase, limit=0):

        self.collection = DB.get_collection('novels_sources_targets')
        self.cursor = self.collection.find({'phase': phase}, limit=limit)
        self.limit = limit

    def __getitem__(self, item):
        document = self.cursor[item]
        source_indexed = document['text_indexed']['source']
        target_indexed = document['text_indexed']['target']
        return source_indexed, target_indexed

    def __iter__(self):
        self.cursor.rewind()
        for document in self.cursor:
            source_indexed = document['text_indexed']['source']
            target_indexed = document['text_indexed']['target']
            yield source_indexed, target_indexed
        self.cursor.rewind()

    def __len__(self):
        if self.limit > 0:
            return self.limit
        else:
            return self.cursor.count()

    @staticmethod
    def prepare_dataset(dictionary):

        # for phase in ['train', 'val', 'test']:
        source_collection = DB.get_collection('novels_sources_targets')
        data = source_collection.find()
        for sentence_document in tqdm(data):
            source_sentences = sentence_document['text']['source']
            target_sentence = sentence_document['text']['target']

            inputs_indexed = [dictionary.index_sentence(source_sentence) for source_sentence in source_sentences]
            targets_indexed = dictionary.index_sentence(target_sentence)
            sentence_document_update = {
                "$set": {
                    'text_indexed': {'source': inputs_indexed,
                                     'target': targets_indexed}}
            }
            source_collection.update_one({'_id': sentence_document['_id']}, sentence_document_update)
