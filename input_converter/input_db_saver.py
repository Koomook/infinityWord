import pymongo
from pymongo import MongoClient
from datetime import datetime
import urllib.request
import re
import json


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

class BooksSaver(object):
    def __init__(self, database_name, collection_name):
        self.client = MongoClient('mongodb://localhost:27272',
                     username='rapper',
                     password='poza0705',
                     authSource='raplab',
                     authMechanism='SCRAM-SHA-1')
        self.db = self.client.get_database(database_name)
        self.c = self.db.get_collection(collection_name)
        self.sbi = SearchBookInfo()
        
    def insert_by_isbn(self, isbn, file_path, ebook=False):
        result = self.sbi.get_basic(isbn)
        item = result['items'][0]
        item['isbn'] = re.sub(' ', '-', re.sub('<b>|</b>','',  item['isbn']))
        item['_id'] = item['isbn']
        
        with open(file_path) as fp:
            txt = fp.read()
            print(txt[:100])
        
        item.update({'text': txt, 'ebook': ebook})

        r = self.c.insert_one(item)
        if r.acknowledged :
            print('>> Insert Success : {}'.format(r.inserted_id))

class LyricsSaver(object):
    # To-do: refactoring jupyter notebook code
    pass

