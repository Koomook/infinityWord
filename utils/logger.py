import logging
import logging.handlers
import pymongo

def create_logger(logging_name, logging_fp=None):
    logger = logging.getLogger(logging_name)
    # levels : debug > info > warning > error > critical

    formatter = logging.Formatter('%(asctime)s [%(levelname)s|%(filename)s:%(lineno)s] > %(message)s')
    formatter_simple = logging.Formatter('|%(levelname)s|%(message)s')
    logger.handlers = []

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter(formatter_simple)
    logger.addHandler(streamHandler)

    if logging_fp != None:

        file_max_bytes = 1 * 1024 * 1024 
        fileHandler = logging.handlers.RotatingFileHandler(filename=logging_fp, maxBytes=file_max_bytes)
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.DEBUG)
        
        logger.addHandler(fileHandler)

    return logger


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

