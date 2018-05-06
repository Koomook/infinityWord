import re
import os, sys
from konlpy.tag import Mecab
from slackclient import SlackClient
import logging
import logging.handlers
from ..utils.logger import FileLogger, MongoLogger



class Processor(object):

    def __init__(self, name, logging_path):
        # Patterns & Lenghth limit
        self.P_CHAPTER = '\n[제 ]*[\d]{1,3}[ ]*[장편막부]+ .{0,50}\n|\n[ ]*[\d]{1,3}[\. ]+.{0,50}\n|\n\n[가-힣]+\n\n|\n[ ]*[\d]{1,3}[ ]*\n|@ff|<title>.*$'
        self.P_WHITESPACE = '\n|\r|\t|\\u200a|\\uf000|\\ufeff|\\u3000'

        self.P_PARANTHESIS = '\(.+?\)|\[.+?\]|[\d]+\)|〔.+?\〕' # 부연 및 주석 패턴 e.g. (blah), [1], 1) 
        self.P_DIALOGUE = '".+?"|\'.+?\'|<talk>.*$' # 대화 및 독백 패턴 e.g. "blah", 'blah'

        self.P_SYMBOL = '[\'\"‘’“”`,·:<>〈〉「」『』《》\+\-─=―_\*]'
        self.P_SUB_SPACE = '(– )|(- )|(─ )' # 스페이스와 대체할 패턴
        self.P_SUB_PERIOD = '…|\.+'

        self.P_DELIMITER = '\.|!|\?'

        self.P_NOT_KOREAN = '[^가-힣0-9 ⎡⎜]'
        self.P_NOT_KorEngNum = '[^가-힣0-9A-Za-z⎡⎜]'

        self.MIN_LEN_CHAPTER = 500
        self.MIN_LEN_CHUNK = 3
        
        self.logger = FileLogger(name, logging_path).get()
        self.tagger = Mecab()

    def process_general(self, book):
        """
        book : string type
        """
        book_processed = []
        for i, chap in enumerate(self.divide_chapter(book)):
            chap = self.skip_linespacing(chap)
            chap = self.skip_elaborate(chap)
            chap = self.process_dialogue(chap)
            chap = self.skip_symbols(chap)
            sen_list = self.split_sentece(chap)
            # self.logger.info('CHAPTER {:2}'.format(i+1))
            for sen in sen_list:
                not_korean = re.findall(self.P_NOT_KOREAN, sen)
                if not_korean != []:
                    self.logger.debug('REQUIRE REVIEW: PART {:2}|{}|{}'.format(i+1, not_korean, sen))
            self.logger.info('PART {:2}: {} lines, {} characters'.format(i+1, len(chap), len(sen_list)))
            book_processed.append(sen_list)
        return book_processed

    # lyrics와 text에서 공통적으로 사용할 수 있는 db_saver 만들기
    # inference의 output_db_saver와는 분리하는 것이 좋을 듯
    def save_to_db():
        pass


    def divide_chapter(self, txt):
        self.logger.debug('PART DELIMITERS: {}'.format(re.findall(self.P_CHAPTER, txt)))
        chapters = re.split(self.P_CHAPTER, txt)
        
        result = []
        for i, chapter in enumerate(chapters):
            self.logger.info('PART {:2}: {}'.format(i+1, len(chapter)))
            if len(chapter) > self.MIN_LEN_CHAPTER:
                result.append(chapter)
            else:
                self.logger.debug('PART {:2} SKIPPED: {}'.format(i+1, chapter))
        return result

    def skip_linespacing(self, txt): # 이름 바꾸기
        # 글의 폭을 맞추기 위한 개행 제거, 불필요한 여백제거
        # 텍스트가 고정폭으로 잘려있지 않아도 이걸 해주면 좋을 것 같음
        txt = re.sub(self.P_WHITESPACE, '', txt)
        return txt

    def skip_elaborate(self, txt): #부연, 주석표시 등 삭제
        return re.sub(self.P_PARANTHESIS, '', txt)

    def skip_symbols(self, txt):
        txt = re.sub(self.P_SYMBOL, '', txt)
        txt = re.sub(self.P_SUB_PERIOD, '.', txt)
        return txt


    def filter_korean(self):
        pass

    def process_dialogue(self, txt):
        # unite adjascent short chunks recursively
        def unite_short_chunks(start, _list):
    #         print(start, _list)
            for i in range(start, len(_list)-1):
                if(len(_list[i])<self.MIN_LEN_CHUNK or len(_list[i+1])<self.MIN_LEN_CHUNK):
                    _list[i] = ''.join([_list[i], _list.pop(i+1)])
                    _list = unite_short_chunks(i, _list)
                    break
            return _list
        
        for dialogue in re.findall(self.P_DIALOGUE, txt):
            # quote 삭제, 필요 없는 구두점을 모두 바꾼 후
            _dialogue = self.skip_symbols(dialogue)
            
            # 하나의 대사 내에서 DELIMITER로 문장을 나누고, ''와 ' '는 제외함
            _list = self.split_sentece(_dialogue)
            
            # 대사를 여러 문장으로 나누었을 때, 두 문장 이상이면 짧은 문장을 묶는다.
            if len(_list) > 1:
                _list = unite_short_chunks(0, _list)
                _dialogue  = '⎡' + '.⎜'.join(_list) + '.'
            
            txt = txt.replace(dialogue, _dialogue, )
            
        return txt

    def split_sentece(self, txt):
        """ DELIMITER로 문장을 나누고, ''와 ' '는 제외함
        
            return: 문장들의 리스트
        """
        _list = [sen.strip(' ') for sen in re.split(self.P_DELIMITER, txt)] 
        _list = list(filter(('').__ne__, _list))
        return _list

    
    def pos_tagging_chapter(self, sen_list):
        chap_tagged = []
        for i, line in enumerate(sen_list):
            tmp_line = []
            for tu in self.tagger.pos(line):
                if tu[0] == '⎡' or tu[0] == '⎜':
                    tmp_line.append(tu[0])
                else:
                    tmp_line.append('/'.join(tu))

                if tu[1] == 'UNKNOWN':
                    self.logger.debug('UNKNOWN TAG: {}'.format('/'.join(tu)))
            chap_tagged.append(' '.join(tmp_line))
        return chap_tagged

    def pos_tagging(self, processed):
        tagged = []
        for chap in processed:
            tagged.append(self.pos_tagging_chapter(chap))
        return tagged

        

