from pymongo import MongoClient
from pymongo.errors import PyMongoError

def connect_mongo(address, username, password, database_name, test=True):
    """connect mongodb and returns database"""
    def access_test():
        db.collection_names()
    try:
        client = MongoClient(host=address,
                                     username=username, 
                                     password=password, 
                                     authSource=database_name, 
                                     authMechanism='SCRAM-SHA-1')
        db = client.get_database(database_name)
    except PyMongoError:
        print('server error')

    if test:
        access_test()
        print('DB Authentication Succeeded')
    return db

def connect_mongo_config(db_config):
    """connect mongodb and returns database"""
    try:
        client = MongoClient(host=db_config['host'],
                             username=db_config['username'],
                             password=db_config['password'],
                             authSource=db_config['authSource'],
                             authMechanism=db_config['authMechanism'])
        db = client.get_database(db_config['authSource'])
        print(db.collection_names())
        return db

    except PyMongoError:
        print('server error')

class GetCollection():

    def __init__(self, address, username, password, database_name, test=True):

        self.db = connect_mongo(address, username, password, database_name, test)

    def novel(self, collection_name='books_converted', limit=0):
        """return cur"""
        collection = self.db.get_collection(collection_name)
        return collection.find().limit(limit)

    def lyrics(self, collection_name='processed', limit=0, keyword=None, korean=True, tags=True):
        """get all korean lyrics documents from mongo db

        Args : 
        
        """
        collection = self.db.get_collection(collection_name)

        query = {}
        if korean == True:
            query['korean'] = True
        elif korean == False:
            query['korean'] = False
        if tags:
            query['tags'] = {'$ne' : []}
        if collection=='processed' and keyword!=None:
            pattern = re.compile('(' + ')|('.join(keyword) + ')')
            query['lyrics.processed.mecab'] = {'$elemMatch': {'$elemMatch' : {'$all' : [pattern]}}}
            
        proj = {'title':True, 'genre':True, 'tags':True, 'lyrics.processed.mecab':True}
        print(query, proj)

        return collection.find(query, proj).limit(limit)

