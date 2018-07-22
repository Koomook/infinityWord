import logging
from datetime import datetime
import os
from os.path import dirname, abspath, join, exists


BASE_DIR = dirname(abspath(__file__))


def get_logger(log_name='log'):

    log_dir = join(BASE_DIR, 'logs', log_name)
    if not exists(log_dir):
        os.mkdir(log_dir)

    log_filename = '{log_name}-{datetime}.log'.format(log_name=log_name, datetime=datetime.now())
    log_filepath = join(log_dir, log_filename)

    logger = logging.getLogger(log_name)
    if not logger.handlers:  # only if logger doesn't already exist
        file_handler = logging.FileHandler(filename=log_filepath, mode='w', encoding='utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

    return logger
