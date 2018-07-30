""" Doesn't Work """

from .text_converter import Converter
from pymongo import MongoClient
from pymongo.errors import DocumentTooLarge, DuplicateKeyError, WriteError
from datetime import datetime
from os.path import dirname, abspath, join

client = MongoClient(host='mongodb://aihub.pozalabs.com:30877',
                     username='rapper',
                     password='poza0705',
                     authSource='raplab',
                     authMechanism='SCRAM-SHA-1')
db = client.get_database('raplab')

txt_type = 'N'
league = 2
logging_name = 'webNovel_convert'
base_dir = dirname(dirname(abspath(__file__)))
logging_path = join(base_dir, 'logs', logging_name, '{}_type{}_league{}.log'.format(datetime.today().strftime("%Y-%m-%d_%H:%M"), txt_type, league))

tp = Converter(txt_type, logging_name=logging_name, logging_path=logging_path)

c_raw = db.get_collection('novels_raw')
c_converted = db.get_collection('novels_converted')

failed = []
for i, book in enumerate(c_raw.find({'$and': [{'type': txt_type}, {'league': league}]})):
    tp.logger.info('***{} id:{} title:{}***'.format(i, book['_id'], book['title']))
    cleaned = tp.convert_to_only(book['text'], pattern=tp.P_NOT_KorEngNum)
    mecab = tp.pos_tagging_book(cleaned)
    try:
        doc = {'_id': book['_id'],
               'genre': book['genre'],
               'title': book['title'],
               'text': {'cleaned': cleaned,
                        'tagged': {'mecab': mecab}
                        },
               'author': book['author'],
               'type': book['type'],
               'league': book['league'],
               'num_volumes': book['num_volumes'],
               'num_stars': book['num_stars'],
               'num_concern': book['num_concern']
               }

        r = c_converted.insert_one(doc)
        if r.acknowledged:
            tp.logger.info('>> Insert Success: {}'.format(r.inserted_id))
        else:
            failed.append(doc)
            tp.looger.info('>> Insert Fail: WRITE ERROR: {}'.format(doc['_id']))

    except DuplicateKeyError as e:
        r = c_converted.update_one({'_id': book['_id']},
                                   {'$set': {'text.cleaned': cleaned,
                                             'text.tagged.mecab': mecab}
                                    }
                                   )
        if r.modified_count:
            tp.logger.info('>> Update Success: {}'.format(book['_id']))
        else:
            tp.logger.info('>> Update Fail: {}'.format(book['_id']))

    except DocumentTooLarge as e:
        failed.append(doc)
        tp.logger.fatal('>> Insert Fail: {}: {}'.format(e, doc['_id']))
#     break