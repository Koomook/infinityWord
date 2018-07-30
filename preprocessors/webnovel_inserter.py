""" Doesn't Work """

import os, sys
from pymongo import MongoClient
from pymongo.errors import DocumentTooLarge, DuplicateKeyError, WriteError
import pickle
from os.path import dirname, abspath, join

client = MongoClient(host='mongodb://aihub.pozalabs.com:30877',
                     username='rapper',
                     password='poza0705',
                     authSource='raplab',
                     authMechanism='SCRAM-SHA-1')
db = client.get_database('raplab')


#### Type of books
# E : e-book (purchased)
# D : Downloaded text file from web
# N : Naver Web Novel crawling

#### leagues of Naver Web Novels (N)
#0: 오늘의웹소설
#1: 베스트,
#2: 챌린지

base_dir = dirname(dirname(abspath(__file__)))
data_dir = join(base_dir, 'data', 'webnovel_data')

# meta_dict
# {'title': ['author', novelid, num_volumes, num_stars, num_concerns]}

with open(join(data_dir, 'metadata_dict_challenge.pickle'), 'rb') as p:
    meta = pickle.load(p)

# Naver Web Novels
# league 구분 - 0:오늘의웹소설, 1:베스트, 2:챌린지 (종류에따라 직접 입력)

c_raw = db.get_collection('novels_raw')

for sub in os.listdir(data_dir):
    sub_path = os.path.join(data_dir, sub)
    if os.path.isdir(sub_path):
        genre = sub
        for file_name in os.listdir(sub_path):
            try:
                with open(os.path.join(sub_path, file_name)) as fp:
    #                 title = re.sub('<subject>|\n', '', fp.readline())
                    text = fp.read()
                title = file_name[:-4]
                meta_row = meta[title]
                doc = {'_id': meta_row[1],
                       'author': meta_row[0],
                       'genre': genre,
                       'title': title,
                       'text': text,
                       'type': 'N',
                       #---------league-------#
                       'league' : 2,
                       #----------------------#
                       'num_volumes': meta_row[2],
                       'num_stars': meta_row[3],
                       'num_concern': meta_row[4]
                }
                r = c_raw.insert_one(doc)
                if r.acknowledged :
                    print('>> Insert Success : {} {}'.format(r.inserted_id, title))
            except KeyError:
                print('>> Key Error : {}'.format(title))
            except DuplicateKeyError:
                print('>> {} {} is already inserted'.format(meta_row[1], title))
            except WriteError as e:
                print('>> {} {} : {}'.format(meta_row[0], title, e))