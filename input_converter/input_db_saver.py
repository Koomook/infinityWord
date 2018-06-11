import os, sys
import pymongo
from pymongo.errors import DocumentTooLarge, DuplicateKeyError, WriteError
import urllib.request
import re
import json
import numpy as np

from ..utils.mongo import connect_mongo_config
from ..utils.logger import create_logger


class SearchBookInfo(object):
    """Naver Search API for books
    (https://developers.naver.com/docs/search/book/)
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
    """Find information about books and insert it with raw text.
    No implimentation for the DuplicateKeyError. check database.

    Args:
        db_config: database configuration
        collection_name: collection name to save new document
    """

    def __init__(self, db_config, collection_name):
        self.db = connect_mongo_config(db_config)
        self.c_raw = self.db.get_collection(collection_name)
        self.sbi = SearchBookInfo()
        self.logger = create_logger('insetrawtext', '')
        
    def insert_by_isbn(self, isbn, fp_or_str, book_type):
        result = self.sbi.get_basic(isbn)
        doc = result['items'][0]
        doc['isbn'] = re.sub(' ', '-', re.sub('<b>|</b>','',  doc['isbn']))
        doc['_id'] = doc['isbn']
        
        if os.path.isdir(fp_or_str):
            with open(file_path) as fp:
                txt = fp.read()
                print(txt[:100])
        else:
            txt = fp_or_str
        
        doc.update({'text': txt, 'type': book_type})
        try:
            r = self.c_raw.insert_one(doc)
            if r.acknowledged :
                print('>> Insert Success : {}'.format(r.inserted_id))
        except:
            print('>> Insert Fail: {}'.format(doc['_id']))


class ConvertSaver(object):
    """To help convert raw data and save them to database

    Args:
        db_config: database to load raw data and save converted data
    """

    def __init__(self, db_config):
        self.db = connect_mongo_config(db_config)

    def check_load_cursor(self, tc, load_cursor):
        if not isinstance(load_cursor, pymongo.cursor.Cursor):
            tc.logger.warning("Please give pymongo `cursor` (Don't use `find_one` method`)")
            raise pymongo.Error.OperationFailure
        if not load_cursor.alive:
            tc.logger.info("cursor rewind")
            load_cursor = load_cursor.rewind()
        
        num_data = load_cursor.count()
        if num_data == 0:
            tc.logger.warning('No document matched')
            raise pymongo.Error.OperationFailure
        else:
            tc.logger.info('{} documents are matched to query'.format(num_data))
        return load_cursor

    def convert_raw(self, tc, load_cursor, save_c_name, split_max_length, test=False):
        """Clean raw data, Tag cleaned data, Split tagged data at once.
        And insert or update database

        Attriibutes:
            tc: Text converter
            load_cursor: Mongodb cursor to iterate documents
            save_c_name : Collection name to save converted data
            split_max_length: criteria length to cut long sentences 
            test: Return listif true, . if false, insert or update on database
        """
        load_cursor = self.check_load_cursor(tc, load_cursor)

        # convert
        if test:
            cleaned_books, tagged_books, split_books = [], [], []
        else:
            c_converted = self.db.get_collection(save_c_name)

        for doc in load_cursor:
            _id, title = doc['_id'], doc['title']
            tc.logger.info('*****{} {}*****'.format(_id, title))
            cleaned = tc.clean_raw(doc['text'], tc.F_KorEngNum)
            tagged = tc.pos_tagging_book(cleaned)
            split = tc.split_long_sentence_book(tagged, split_max_length, False)
            
            if test:
                cleaned_books.append(cleaned)
                tagged_books.append(tagged)
                split_books.append(split)

            else: # Not test, insert new or update exsting doc
                doc['text'] = {
                    'cleaned': cleaned,
                    'tagged': {'mecab': tagged},
                    'split': {'mecab': split}
                }

                # insert or update
                try:
                    r = c_converted.insert_one(doc)
                    tc.logger.info('>> Insert Success: {}'.format(_id))
                except DuplicateKeyError as e:
                    tc.logger.info('>> Insert Fail: {}'.format(e))
                    r = c_converted.update_one(
                            {'_id': _id},
                            {'$set': {'text.cleaned': cleaned,
                                      'text.tagged.mecab': tagged,
                                      'text.split.mecab': split}}
                        )

                    if r.raw_result['ok']:
                        tc.logger.info('>> Update Sucess: {}'.format(_id))
                    else:
                        tc.logger.warning('>> Update Fail: {} {}\n: {}'.format(_id, title, r.raw_result))
                except DocumentTooLarge as e:
                    tc.logger.warning('>> Insert Fail: {} {}\n: {}'.format(_id, title, e))

        load_cursor.close()

        if test: 
            return cleaned_books, tagged_books, split_books

    def tag_cleaned(self, tc, load_cursor, split_max_length, test=False):
        """Tag POS to cleaned data and update database.

        Args:
            tc: text converter
            load_cursor: mongodb cursor to iterate documents
        """

        load_cursor = self.check_load_cursor(tc, load_cursor)

        if test:
            tagged_books, split_books = [], []
        else:
            c_converted = load_cursor.collection
        total = load_cursor.count()
        updated = 0

        for doc in load_cursor:
            _id, title = doc['_id'], doc['title']

            tc.logger.info('*****{} {}*****'.format(_id, title))
            cleaned = doc['text']['cleaned']
            tagged = doc['text']['tagged']['mecab']
            new_tagged = tc.pos_tagging_book(cleaned)

            if tagged == new_tagged:
                tc.logger.info('>> No Changes in tagged data: {} {}'.format(_id, title))
            else:
                split = tc.split_long_sentence_book(tagged, split_max_length, False)
                if test:
                    tagged_books.append(new_tagged)
                    split_books.append(split)
                else:
                    try:
                        r = c_converted.update_one(
                            {'_id': _id},
                            {'$set': {'text.tagged.mecab': new_tagged,
                                      'text.split.mecab': split}},
                            upsert=True
                        )
                        if r.raw_result['ok']:
                            tc.logger.info('>> Update Sucess: {} {}'.format(_id, title))
                            updated += 1
                        else:
                            tc.logger.warning('>> Update Fail: {} {}\n{}'.format(_id, title, r.raw_result))
                    except WriteError as e:
                        tc.logger.warning('>> Updated Success: Write Error{}'.format(e))

        tc.logger.info('\ntotal doc: {}\nupdated_doc: {}'.format(total, updated))
        load_cursor.close()

        if test:
            return tagged_books, split_books

    def split_tagged(self, tc, load_cursor, split_max_length, test=False):
        """Split long sentneces from tagged data and update database.
        
        Args:
            tc: text converter
            load_cursor: mongodb cursor to iterate documents
            split_max_length: criteria length to cut sentence
        """
        load_cursor = self.check_load_cursor(tc, load_cursor)

        if test:
            split_books = []
        else:
            c_converted = load_cursor.collection
        total = load_cursor.count()
        updated = 0

        for doc in load_cursor:
            _id, title = doc['_id'], doc['title']

            tc.logger.info('*****{} {}*****'.format(_id, title))
            tagged = doc['text']['tagged']['mecab']
            split_book = tc.split_long_sentence_book(tagged, split_max_length, False)

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
                        tc.logger.info('>> Updated Success: {} {}\n'.format(_id, title))
                    else:
                        tc.logger.warning('>> Update Fail: {} {}\n{}'.format(_id, title, r.raw_result))
                except WriteError as e:
                    tc.logger.warning('>> Updated Success: Write Error{}'.format(e))
        load_cursor.close()

        if test:
            return split_books
