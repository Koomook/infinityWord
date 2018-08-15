from pymongo import MongoClient
from random import seed, random, shuffle
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


class ChaptersDataset:

    def __init__(self, phase, num_novels=2):

        self.books_converted = DB.get_collection('books_converted')
        self.cursor = self.books_converted.find(filter={},
                                           projection={'title': True, 'text.cleaned': True},
                                           limit=num_novels)

        collection = DB.get_collection('novels_chapters')
        self.cursor = collection.find({'phase': phase})

    def __getitem__(self, index):
        chapter = self.cursor[index]
        return chapter['text']

    def __len__(self):
        return self.books_converted.estimated_document_count()

    @staticmethod
    def prepare_dataset(num_novels=2):
        books_converted = DB.get_collection('books_converted')
        cursor = books_converted.find(filter={},
                                      projection={'title': True, 'text.cleaned': True},
                                      limit=num_novels)
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        print(train_ratio + val_ratio + test_ratio)

        chapters_collection = DB.get_collection('novels_chapters')
        chapters_collection.drop()

        seed(0)
        for novel_index, novel in enumerate(cursor):
            for chapter_index, chapter in enumerate(novel['text']['cleaned']):
                chapter_document = {
                    'novel_id' : novel['_id'],
                    'chapter_index' : chapter_index,
                    'title': novel['title'],
                    'text': chapter
                }

                random_number = random()
                if random_number < train_ratio:
                    chapter_document['phase'] = 'train'
                    chapters_collection.insert_one(chapter_document)
                elif random_number < train_ratio + val_ratio:
                    chapter_document['phase'] = 'val'
                    chapters_collection.insert_one(chapter_document)
                else:
                    chapter_document['phase'] = 'test'
                    chapters_collection.insert_one(chapter_document)

        cursor.close()


class ChaptersDatasetOnTheFly:

    def __init__(self, phase, num_novels=2):
        books_converted = DB.get_collection('books_converted')
        self.cursor = books_converted.find(filter={},
                                           projection={'title': True, 'text.cleaned': True},
                                           limit=num_novels)

        all_chapter_locations = [(novel_index, chapter_index)
                                  for novel_index, novel in enumerate(self.cursor)
                                  for chapter_index, chapter in enumerate(novel['text']['cleaned'])]

        self.cursor.rewind()

        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        seed(0)
        data_size = len(all_chapter_locations)
        train_size = int(data_size * train_ratio)
        val_size = int(data_size * val_ratio)

        shuffle(all_chapter_locations)
        train_data = all_chapter_locations[:train_size]
        val_data = all_chapter_locations[train_size:train_size + val_size]
        test_data = all_chapter_locations[train_size + val_size:]

        if phase == 'train':
            self.chapter_locations = train_data
        elif phase == 'val':
            self.chapter_locations = val_data
        elif phase == 'test':
            self.chapter_locations = test_data
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        novel_index, chapter_index = self.chapter_locations[index]
        novel = self.cursor[novel_index]
        return novel['text']['cleaned'][chapter_index]

    def __len__(self):
        return len(self.chapter_locations)


class SentencesDataset:

    def __init__(self, phase):

        assert phase in ('train', 'val', 'test')
        self.collection = DB.get_collection('novels_sentences')
        self.cursor = self.collection.find({'phase': phase})

    def __getitem__(self, index):
        sentence_tokenized = self.cursor[index]
        return sentence_tokenized['text']

    def __len__(self):
        return self.collection.estimated_document_count()

    def __iter__(self):
        for sentence_tokenized in self.cursor:
            yield sentence_tokenized['text']
        self.cursor.rewind()

    @staticmethod
    def prepare_dataset():

        source = DB.get_collection('novels_chapters')
        target = DB.get_collection('novels_sentences')
        target.drop()

        cursor = source.find().batch_size(10)
        for chapter in tqdm(cursor):
            chapter_text = chapter['text']
            for sentence_index, sentence in enumerate(chapter_text):
                sentence_document = {
                    'novel_id': chapter['novel_id'],
                    'chapter_index': chapter['chapter_index'],
                    'sentence_index': sentence_index,
                    'text': sentence,
                    'phase': chapter['phase']
                }
                target.insert_one(sentence_document)

        cursor.close()


class ChaptersTokenizedDataset:

    def __init__(self, phase):
        self.collection = DB.get_collection('novels_chapters')
        self.cursor = self.collection.find({'phase':phase})

    def __getitem__(self, index):
        chapter = self.cursor[index]
        return chapter['text_tokenized']

    def __len__(self):
        return self.collection.estimated_document_count()

    @staticmethod
    def prepare_dataset(tokenizer=None):
        source = DB.get_collection('novels_chapters')
        cursor = source.find({}).batch_size(10)
        for chapter in tqdm(cursor):
            chapter_text = chapter['text']
            if tokenizer is not None:
                chapter_tokenized = tokenizer.tokenize_chapter(chapter_text)
            else:
                chapter_tokenized = [sentence.split() for sentence in chapter_text]
            chapter_document_update = {
                "$set": {
                    'text_tokenized': chapter_tokenized
                }
            }
            source.update_one({'_id': chapter['_id']}, chapter_document_update)
        cursor.close()


class SentencesTokenizedDataset:

    def __init__(self, phase):

        assert phase in ('train', 'val', 'test')
        self.collection = DB.get_collection('novels_sentences')
        self.cursor = self.collection.find({'phase': phase})

    def __getitem__(self, index):
        sentence_tokenized = self.cursor[index]
        return sentence_tokenized['text_tokenized']

    def __len__(self):
        return self.collection.estimated_document_count()

    def __iter__(self):
        for sentence_tokenized in self.cursor:
            yield sentence_tokenized['text_tokenized']
        self.cursor.rewind()


    @staticmethod
    def prepare_dataset():
        source = DB.get_collection('novels_chapters')
        target = DB.get_collection('novels_sentences')
        target.drop()

        cursor = source.find().batch_size(10)
        for chapter in tqdm(cursor):

            for sentence_index, (sentence, sentence_tokenized) in enumerate(zip(chapter['text'], chapter['text_tokenized'])):
                sentence_document = {
                    'novel_id': chapter['novel_id'],
                    'chapter_index': chapter['chapter_index'],
                    'sentence_index': sentence_index,
                    'text': sentence,
                    'text_tokenized': sentence_tokenized,
                    'phase': chapter['phase']
                }
                target.insert_one(sentence_document)
        cursor.close()


class SentencesTokenizedDatasetOnTheFly:

    def __init__(self, phase):
        self.source = ChaptersTokenizedDataset(phase)
        self.sentence_locations = [(chapter_index, sentence_index)
                                   for chapter_index, chapter in enumerate(self.source)
                                   for sentence_index, sentence in enumerate(chapter)]

    def __getitem__(self, index):
        chapter_index, sentence_index = self.sentence_locations[index]
        sentence_tokenized = self.source[chapter_index][sentence_index]
        return sentence_tokenized

    def __len__(self):
        return len(self.sentence_locations)


class InputTargetDataset:
    # TODO: Add bptt arg

    def __init__(self, phase):
        self.source = SentencesTokenizedDataset(phase)

    def __getitem__(self, item):
        sentence = self.source[item]
        inputs, targets = self.process(sentence)
        return inputs, targets

    def __len__(self):
        return len(self.source)

    @staticmethod
    def process(sentence):
        sentence_start_end = [START_TOKEN] + sentence + [END_TOKEN]
        inputs = sentence_start_end[:-1]
        targets = sentence_start_end[1:]
        return inputs, targets


class InputTargetIndexedDataset:

    def __init__(self, phase):

        self.collection = DB.get_collection('novels_sentences')
        self.cursor = self.collection.find({'phase': phase})

    def __getitem__(self, item):
        document = self.cursor[item]
        input_indexed = document['input_target_indexed']['input']
        target_indexed = document['input_target_indexed']['target']
        return input_indexed, target_indexed

    def __len__(self):
        return self.collection.estimated_document_count()

    @staticmethod
    def prepare_dataset(dictionary):

        source = DB.get_collection('novels_sentences')
        cursor = source.find().batch_size(10)
        for sentence in tqdm(cursor):
            inputs, targets = InputTargetDataset.process(sentence['text_tokenized'])
            inputs_indexed = dictionary.index_sentence(inputs)
            targets_indexed = dictionary.index_sentence(targets)
            sentence_document_update = {
                "$set": {
                    'input_target_indexed': {
                        'input': inputs_indexed,
                        'target': targets_indexed
                    }
                }
            }
            source.update_one({'_id': sentence['_id']}, sentence_document_update)
        cursor.close()


class InputTargetIndexedDatasetOnTheFly:

    def __init__(self, phase, dictionary):
        self.source = InputTargetDataset(phase)
        self.dictionary = dictionary

    def __getitem__(self, item):
        inputs, targets = self.source[item]
        inputs_indexed = [self.dictionary[word] for word in inputs]
        targets_indexed = [self.dictionary[word] for word in targets]

        return inputs_indexed, targets_indexed

    def __len__(self):
        return len(self.source)