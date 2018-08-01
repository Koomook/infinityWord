from pymongo import MongoClient
from random import seed, random, shuffle
from tqdm import tqdm
import json

START_TOKEN = '<StartSent>'
END_TOKEN = '<EndSent>'
MONGODB_CONFIG = json.load(open('mongodb_config.json'))

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

        if phase == 'train':
            train_chapters = db.get_collection('novels_train_chapters')
            self.cursor = train_chapters.find()
        elif phase == 'val':
            val_chapters = db.get_collection('novels_val_chapters')
            self.cursor = val_chapters.find()
        elif phase == 'test':
            test_chapters = db.get_collection('novels_test_chapters')
            self.cursor = test_chapters.find()
        else:
            raise NotImplementedError()

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

        train_chapters_collection = DB.get_collection('novels_train_chapters')
        val_chapters_collection = DB.get_collection('novels_val_chapters')
        test_chapters_collection = DB.get_collection('novels_test_chapters')

        # Empty collections
        train_chapters_collection.remove({})
        val_chapters_collection.remove({})
        test_chapters_collection.remove({})

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
                    train_chapters_collection.insert_one(chapter_document)
                elif random_number < train_ratio + val_ratio:
                    val_chapters_collection.insert_one(chapter_document)
                else:
                    test_chapters_collection.insert_one(chapter_document)


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


class ChaptersTokenizedDataset:

    def __init__(self, phase):
        self.source = ChaptersDataset(phase)

    def __getitem__(self, index):
        chapter = self.source[index]
        chapter_tokenized = self.tokenize_chapter(chapter)
        return chapter_tokenized

    def __len__(self):
        return len(self.source)

    @staticmethod
    def tokenize_chapter(chapter):
        chapter_tokenized = []
        for sentence in chapter:
            sentence_tokenized = sentence.split()
            chapter_tokenized.append(sentence_tokenized)
        return chapter_tokenized


class SentencesTokenizedDataset:

    def __init__(self, phase):

        assert phase in ('train', 'val', 'test')
        self.collection = DB.get_collection('novels_' + phase + '_sentences')
        self.cursor = self.collection.find()

    def __getitem__(self, index):
        sentence_tokenized = self.cursor[index]
        return sentence_tokenized['text']

    def __len__(self):
        return self.collection.estimated_document_count()

    def __iter__(self):
        self.cursor.rewind()
        for sentence_tokenized in self.cursor:
            yield sentence_tokenized['text']

    @staticmethod
    def prepare_dataset():

        for phase in ['train', 'val', 'test']:
            source = DB.get_collection('novels_' + phase + '_chapters')
            target = DB.get_collection('novels_' + phase + '_sentences')
            target.remove({})

            for chapter in tqdm(source.find()):
                tokenized_chapter = ChaptersTokenizedDataset.tokenize_chapter(chapter['text'])
                for sentence_index, sentence in enumerate(tokenized_chapter):
                    sentence_document = {
                        'novel_id': chapter['novel_id'],
                        'chapter_index': chapter['chapter_index'],
                        'sentence_index': sentence_index,
                        'text': sentence
                    }
                    target.insert_one(sentence_document)


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

        self.collection = DB.get_collection('novels_' + phase + '_sentences_indexed')
        self.cursor = self.collection.find()

    def __getitem__(self, item):
        document = self.cursor[item]
        inputs_indexed = document['inputs']
        targets_indexed = document['targets']
        return inputs_indexed, targets_indexed

    def __len__(self):
        return self.collection.estimated_document_count()

    @staticmethod
    def prepare_dataset(dictionary):

        for phase in ['train', 'val', 'test']:
            source = DB.get_collection('novels_' + phase + '_sentences')
            target = DB.get_collection('novels_' + phase + '_sentences_indexed')
            target.remove({})

            for sentence in tqdm(source.find()):
                inputs, targets = InputTargetDataset.process(sentence['text'])
                inputs_indexed = InputTargetIndexedDataset.index_sentence(inputs, dictionary)
                targets_indexed = InputTargetIndexedDataset.index_sentence(targets, dictionary)
                sentence_document = {
                    'novel_id': sentence['novel_id'],
                    'chapter_index': sentence['chapter_index'],
                    'sentence_index': sentence['sentence_index'],
                    'inputs': inputs_indexed,
                    'targets': targets_indexed
                }
                target.insert_one(sentence_document)

    @staticmethod
    def index_sentence(sentence, dictionary):
        return [dictionary[word] for word in sentence]


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


if __name__ == '__main__':

    from datetime import datetime
    start_time = datetime.now()

    ChaptersDataset.prepare_dataset(num_novels=2)

    chapters_dataset = ChaptersDataset(phase='train')
    print('chapters_dataset', chapters_dataset[0])

    SentencesTokenizedDataset.prepare_dataset()
    sentences_dataset = SentencesTokenizedDataset(phase='train')
    print('sentences_dataset', sentences_dataset[0])

    from os.path import dirname, abspath
    import sys
    BASE_DIR = dirname(dirname(abspath(__file__)))
    sys.path.append(BASE_DIR)

    from dictionaries import BaseDictionary

    dictionary = BaseDictionary()
    dictionary.prepare_dictionary(sentences_dataset)
    dictionary.save('base_dictionary')

    InputTargetIndexedDataset.prepare_dataset(dictionary=dictionary)
    input_target_indexed_dataset = InputTargetIndexedDataset(phase='val')
    print('input_target_indexed_dataset', input_target_indexed_dataset[0])

    end_time = datetime.now()
    print('start_time', start_time)
    print('end_time', end_time)
    print('took', end_time - start_time)
