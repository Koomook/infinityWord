import logging
import logging.handlers
import pymongo

class FileLogger(object):
    def __init__(self, name, log_file_path):
        self.logger = logging.getLogger(name)
        # levels : debug > info > warning > error > critical
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s|%(filename)s:%(lineno)s] > %(message)s')

        file_max_bytes = 1 * 1024 * 1024 
        file_backupCount = 10
        fileHandler = logging.handlers.RotatingFileHandler(filename=log_file_path, maxBytes=file_max_bytes)
        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.DEBUG)
        streamHandler.setLevel(logging.INFO)

        self.logger.handlers = []
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

    def get(self):
        return self.logger



class MongoLogger(object):
    # 수정 필요
    # pymongo.MongoClient()
    
    def __init__(name, 
                 db='mongolog', 
                 collection='log', 
                 host='localhost', 
                 port=None, 
                 level=logging.NOTSET):
        self.name = name

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(MongoHandler(db, collection, host, port, level))

    def get(self):
        return self.logger    


class MongoHandler(logging.Handler):

    def __init__(self, db, collection, host, port, level):
        logging.Handler.__init__(self, level)
        self.collection = Connection(host, port)[db][collection]

    def emit(self, record):
        data = record.__dict__.copy()
        try:
            self.collection.save(data)
        except InvalidDocument as e:
            logging.error("Unable save log to mongodb: %s", e.message)

