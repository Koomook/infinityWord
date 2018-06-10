import os, sys
import pymongo
from pymongo.errors import DocumentTooLarge, DuplicateKeyError, WriteError
import urllib.request
import re
import json
import numpy as np

from ..utils.mongo import connect_mongo_config


class SearchBookInfo(object):
    """
    https://developers.naver.com/docs/search/book/

    """

    def __init__(self):
        self.client_id = "GgcVvPoU2JsQupPwpKPg"
        self.client_secret = "jowPlwpUKB"

    def get_basic(self, title):
        encText = urllib.parse.quote(title)
        url = "https://openapi.naver.com/v1/search/book?query=" + encText # json 결과
        # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # xml 결과
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)

        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if(rescode==200):
            response_body = response.read()
            response_json = json.loads(response_body.decode('utf-8'))
        else:
            print("Error Code:" + rescode)

        return response_json

class RawBookSaver(object):
    def __init__(self, db_config, database_name, collection_name):
        self.db = connect_mongo_config(db_config)
        self.c = self.db.get_collection(collection_name)
        self.sbi = SearchBookInfo()
        
    def insert_by_isbn(self, isbn, fp_or_str, book_type):
        result = self.sbi.get_basic(isbn)
        item = result['items'][0]
        item['isbn'] = re.sub(' ', '-', re.sub('<b>|</b>','',  item['isbn']))
        item['_id'] = item['isbn']
        
        if os.path.isdir(fp_or_str):
            with open(file_path) as fp:
                txt = fp.read()
                print(txt[:100])
        else:
            txt = fp_or_str
        
        item.update({'text': txt, 'type': book_type})

        r = self.c.insert_one(item)
        if r.acknowledged :
            print('>> Insert Success : {}'.format(r.inserted_id))

class ConvertSaver(object):
    def __init__(self, db_config):
        self.db = connect_mongo_config(db_config)

    def convert_raw(self, tp, load_cursor, save_c_name, split_max_length=None, test=False):
        """Clean raw text and tag cleaned text
        Args:
            tp: text processor
            load_cursor: 
            save_c_name : collection name to save converted data
            test: 
        """
        if not isinstance(load_cursor, pymongo.cursor.Cursor):
            print("Please give pymongo `cursor` (Don't use `find_one` method`)")

        # convert
        if test:
            cleaned_list, tagged_list, split_list = [], [], []
        else:
            self.c_converted = self.db.get_collection(save_c_name)

        for doc in load_cursor:
            tp.logger.info('*****{} {}*****'.format(doc['_id'], doc['title']))
            cleaned = tp.clean_raw(doc['text'], tp.P_NOT_KorEngNum)
            tagged = tp.pos_tagging_book(cleaned)
            split = tp.split_long_sentence_book(tagged, split_max_length, False)
            
            if test:
                cleaned_list.append(cleaned)
                tagged_list.append(tagged)
                split_list.append(split)

            else: # Not test, insert new or update exsting doc
                doc['text'] = {
                    'cleaned': cleaned,
                    'tagged': {'mecab': tagged},
                    'split': {'mecab': split}
                }

                # insert or update
                try:
                    r = self.c_converted.insert_one(doc)
                    tp.logger.info('>> Insert Success: {}'.format(doc['_id']))
                except DuplicateKeyError as e:
                    tp.logger.info('>> Insert Fail: {}'.format(e))
                    r = self.c_converted.update_one(
                            {'_id': doc['_id']},
                            {'$set': {'text.cleaned': cleaned,
                                      'text.tagged.mecab': tagged,
                                      'text.split.mecab': split}}
                        )

                    if r.raw_result['ok']:
                        tp.logger.info('>> Update Sucess: {}'.format(doc['_id']))
                    else:
                        tp.logger.info('>> Update Fail: {}: {}'.format(doc['_id'], r.raw_result))
                except DocumentTooLarge as e:
                    tp.logger.info('>> Insert Fail: {}: {}'.format(doc['_id'], e))

        load_cursor.close()

        if test: 
            return cleaned_list, tagged_list, split_list
        

    def tag_cleaned(self, tp, load_cursor):
        """
        Args:
            cvtd_c_name : collection namme of converted data
        """

        if not isinstance(load_cursor, pymongo.cursor.Cursor):
            print("Please give pymongo `cursor` (Don't use `find_one` method`)")

        if test:
            tagged_books = []
        else:
            c_converted = load_cursor.collection
        total = load_cursor.count()
        updated = 0

        for doc in load_cursor:
            _id, title = doc['_id'], doc['title']

            tp.logger.info('*****{} {}*****'.format(_id, title))
            cleaned = doc['text']['cleaned']
            tagged = doc['text']['tagged']['mecab']
            new_tagged = tp.pos_tagging_book(cleaned)

            if tagged == new_tagged:
                tp.logger.info('>> No Changes in tagged data: {} {}'.format(doc['']))
            else:
                if test:
                    tagged_books.append(new_tagged)
                else:
                    updated += 1
                    r = c_converted.update_one(
                        {'_id': _id},
                        {'$set': {'text.tagged.mecab': new_tagged}},
                        upsert=True
                    )

                    if r.raw_result['ok']:
                        tp.logger.info('>> Update Sucess: {} {}'.format(_id, title))
                    else:
                        tp.logger.info('>> Update Fail: {} {}\n{}'.format(_id, title, r.raw_result))
        tp.logger.info('total doc: {}\nupdated_doc: {}'.format(total, updated))
        load_cursor.close()

        if test:
            return tagged_books


    def split_tagged(self, tp, load_cursor, max_length, test=False):

        if not isinstance(load_cursor, pymongo.cursor.Cursor):
            print("Please give pymongo `cursor` (Don't use `find_one` method`)")

        self.load_cursor = load_cursor
        # if not load_cursor.alive:
        #     load_cursor = load_cursor.rewind()

        if test:
            split_books = []
        else:
            c_converted = load_cursor.collection

        for doc in self.load_cursor:
            _id, title = doc['_id'], doc['title']

            tp.logger.info('*****{} {}*****'.format(_id, title))
            tagged = doc['text']['tagged']['mecab']
            split_book = tp.split_sentence_book(tagged, max_length, False)

            if test:
                split_books.append(split_book)
            else:
                # update doc
                try:
                    r = c_converted.update_one(
                            {'_id': doc['_id']}, 
                            {'$set': {'text.split.mecab': split_book}}, 
                            upsert=True
                        )
                    if r.raw_result['ok']:
                        tp.logger.info('>> Updated Success: {} {}\n'.format(_id, title))
                    else:
                        tp.logger.info('>> Update Fail: {} {}\n{}'.format(_id, title, r.raw_result))
                except WriteError as e:
                    tp.logger.warning('>> Updated Success: Write Error{}'.format(e))

        load_cursor.close()

        if test:
            return split_books
        

class LyricsSaver(object):
    # To-do: refactoring jupyter notebook code
    pass

