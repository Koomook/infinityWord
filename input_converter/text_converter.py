import re
import os, sys
from konlpy.tag import Mecab
from slackclient import SlackClient
from collections import Counter

if not '..' in sys.path:
    sys.path.append('..')
from ..utils.logger import FileLogger, MongoLogger



class Processor(object):

    def __init__(self, name, logging_path, txt_type, mono=True):
        # Patterns & Lenghth limit
        self.logger = FileLogger(name, logging_path).get()
        self.tagger = Mecab()
        self.unknowns = Counter()
        self.mono = mono

        if txt_type == 'N':
            self.P_CHAPTER = '<title>.*\n' 
            self.P_WHITESPACE = '\r|\t|\\u200a|\\uf000|\\ufeff|\\u3000|\\xa0'
            self.P_DELIMITER = '(?<!\d)\.(?![\d]+)|\!|\?|\n' # \n 포함
            self.P_DIALOGUE = '<talk>.*\n|\n *".+?"|\n *\'.+?\'|\n *‘.+?’|\n *“.+?”|\n *「.+?」'
            
        elif txt_type == 'D': # \n 을 먼저 제거하는 과정이 꼭 필요함
            self.P_CHAPTER = '\n[제 ]*[\d]{1,3}[ ]*[장편막부]+ .{0,50}\n|\n[ ]*[\d]{1,3}[\. ]+.{0,50}\n|\n\n[가-힣]+\n\n|\n[ ]*[\d]{1,3}[ ]*\n|@ff'
            self.P_WHITESPACE = '\n|\r|\t|\\u200a|\\uf000|\\ufeff|\\u3000|\\xa0'
            self.P_DELIMITER = '(?<!\d)\.(?![\d]+)|\!|\?'
            self.P_DIALOGUE = '".+?"|\'.+?\'|‘.+?’|“.+?”|「.+?」'

        elif txt_type == 'E':
            self.P_CHAPTER = '\n[제 ]*[\d]{1,3}[ ]*[장편막부]+ .{0,50}\n|\n[ ]*[\d]{1,3}[\. ]+.{0,50}\n|\n\n[가-힣]+\n\n|\n[ ]*[\d]{1,3}[ ]*\n|@ff'
            self.P_WHITESPACE = '\r|\t|\\u200a|\\uf000|\\ufeff|\\u3000|\\xa0'
            self.P_DELIMITER = '(?<!\d)\.(?![\d]+)|\!|\?|\n' 
            self.P_DIALOGUE = '\n *".+?"|\n *\'.+?\'|\n *‘.+?’|\n *“.+?”|\n *「.+?」'
        
        self.P_PARANTHESIS = '\(.+?\)|\[.+?\]|[\d]+\)|〔.+?\〕' # 부연 및 주석 패턴 e.g. (blah), [1], 1) 
        # 삭제해도 무방한 심볼들과 P_PRMT_SYMBOLS 에 해당하는 문자들 중에서 삭제해야하는 케이스를 정의
        self.P_SKIP_SYMBOLS = '<talk>|(?<!\d)~+(?![\d]+)|(?<!\d)%|[\'\"\*\+\-\^\(\)\|‘’“”`;·:〈〉<>「」『』《》【】〖〗─=―—ㅡ_⎯]' 
        
        
        # Radicals, Symbols & Punctuation, Strokes, enclosed letters & Compatibility, extension A, Unified
        # self.U_CJK_hanja = '\\u2E80-\\u2FDF\\u3000-\\u303F\\u31C0-\\u31EF\\u3200-\\u33FF\\u3400-\\u4D8F\\u4E00-4FFF'
        # Hiragana, Katakana, Kanbun, Halfwidth and Fullwidth Forms
        # self.U_Japanese = '\\u3040-\\u309F\\u30A0-\\u31FF\\u3190-\\u319F\\uFF00-\\uFFEF'

        
        self.SD = '🐱' # 문장 구분 (Sentence Delimeter)
        self.D_SYMBOLS = ['"', "'", '‘', '’', '“', '”']
        self.DS1 = '⎡' # 대사의 시작 문장 (Dialogue Start), Double quote
        self.DS2 = '⎛' # 대사의 시작 문장 (Dialogue Start), Single quote
        self.DC = '⎜' # 대사의 이어지는 문장 (Dialogue Continue)
        
        self.P_PRMT_SYMBOLS = '%/&~…\.\?\!,{}{}{}{}'.format(self.SD, self.DS1, self.DS2, self.DC)

        self.P_NOT_KorNum = '[^ 가-힣0-9\\u1100-\\u11FF\\u3130-\\u318F{}]'.format(self.P_PRMT_SYMBOLS)
        self.P_NOT_KorEngNum = '[^ A-Za-z가-힣0-9\\u1100-\\u11FF\\u3130-\\u318F{}]'.format(self.P_PRMT_SYMBOLS)

        self.MIN_LEN_CHAPTER = 500
        self.MIN_LEN_CHUNK = 4 # P_DELIMITER로 구분하였을 때 글자수가 3 이하이면, 주위 chunk에 붙인다
        
    def convert_general(self, book):
        """
        book : string type
        """
        book_cleaned = []
        for i, chap in enumerate(self.divide_chapter(book)):
            chap = self.skip_linespacing(chap)
            chap = self.skip_elaborate(chap)
            chap = self.sub_symbols(chap)
            chap = self.process_dialogue(chap)
            sen_list = self.split_sentece(chap)
            
            for sen in sen_list:
                not_korean = re.findall(self.P_NOT_KorEngNum, sen)
                if not_korean != []:
                    self.logger.debug('REVIEW THIS SENTENCE: PART {:3}: {} <- {}'.format(i+1, not_korean, sen))
            self.logger.info('PART {:3}: {} characters, {} lines'.format(i+1, len(chap), len(sen_list)))
            book_cleaned.append(sen_list)
        return book_cleaned

    def convert_to_only(self, book, pattern):
        book_cleaned = []
        num_line_acc = 0
        for i, chap in enumerate(self.divide_chapter(book)):
            chap = self.skip_linespacing(chap)
            chap = self.skip_elaborate(chap)
            chap = self.sub_symbols(chap)
            chap = self.process_dialogue(chap)
            sen_list = self.split_sentece(chap)            
            
            _sen_list = []
            for sen in sen_list:
                if sen == self.DS1 or sen == self.DS2:
                    self.logger.debug('REVIEW PART {:3}: only dialogue start token'.format(i+1))
                else:
                    _not = re.findall(pattern, sen)
                    if _not != []:
                        if _not in self.D_SYMBOLS:
                            self.logger.debug('-SKIPPED SYMBOLS: PART {:3}: {} <- {}'.format(i+1, _not, sen))
                        sen_only = re.sub(pattern, '', sen)
                        if not sen_only == '':
                            _sen_list.append(sen_only)
                    else:
                        _sen_list.append(sen)
            len_list = len(_sen_list)
            num_line_acc += len_list
            self.logger.info('PART {:3}: {} lines, {} lines accumulated'.format(i+1, len_list, num_line_acc))
            book_cleaned.append(_sen_list)
        return book_cleaned

    def divide_chapter(self, txt):
        self.logger.debug('PART DELIMITERS: {}'.format(re.findall(self.P_CHAPTER, txt)))
        chapters = re.split(self.P_CHAPTER, txt)
        
        result = []
        for i, chapter in enumerate(chapters):
            # self.logger.debug('PART {:2}: {}'.format(i+1, len(chapter)))
            if len(chapter) > self.MIN_LEN_CHAPTER:
                result.append(chapter)
            else:
                self.logger.debug('PART {:3} SKIPPED: {}'.format(i+1, chapter))
        return result

    def skip_linespacing(self, txt): # 이름 바꾸기
        # 글의 폭을 맞추기 위한 개행 제거, 불필요한 여백제거
        # 텍스트가 고정폭으로 잘려있지 않아도 이걸 해주면 좋을 것 같음
        txt = re.sub(self.P_WHITESPACE, '', txt)
        return txt

    def skip_elaborate(self, txt): #부연, 주석표시 등 삭제
        return re.sub(self.P_PARANTHESIS, '', txt)

    def skip_symbols(self, txt):
        txt = re.sub(self.P_SKIP_SYMBOLS, '', txt)
        return txt

    def sub_symbols(self, txt):
        txt = re.sub('\.{2,3}', '…', txt)
        txt = re.sub('…{2,}', '……', txt)
        
        txt = re.sub('(?<!\d)\.(?![\d]+)', '.'+self.SD, txt)
        txt = re.sub('‼+|\!+', '!'+self.SD, txt)
        txt = re.sub('\?+', '?'+self.SD, txt)
        return txt

    def process_dialogue(self, txt):
        # unite adjascent short chunks recursively
        def unite_short_chunks(start, _list):
            for i in range(start, len(_list)-1):
                if(len(_list[i])<self.MIN_LEN_CHUNK or len(_list[i+1])<self.MIN_LEN_CHUNK):
                    _list[i] = ''.join([_list[i], _list.pop(i+1)])
                    _list = unite_short_chunks(i, _list)
                    break
            return _list
        
        for dialogue in re.findall(self.P_DIALOGUE, txt):
            # quote 삭제, 필요 없는 구두점을 모두 바꾼 후, 
            # (구두점이 있으면 split의 길이가 길어져 united 되는 빈도가 적어진다)
            if self.mono == True and (dialogue.strip()[0] in ["'", "‘"]):
                DS = self.DS2
            else:
                DS = self.DS1

            _dialogue = self.skip_symbols(dialogue)
            # _dialogue = self.sub_symbols(dialogue)
            # print(_dialogue)
            
            # 하나의 대사 내에서 DELIMITER로 문장을 나누고, ''와 ' '는 제외함
            _list = self.split_sentece(_dialogue)
            # 대사를 여러 문장으로 나누었을 때, 두 문장 이상이면 짧은 문장을 묶는다.
            if len(_list) > 1:
                # try:
                _list = unite_short_chunks(0, _list)
                # except RecursionError as e:
                #     self.logger.fatal('{}'.format(e))
                #     self.logger.info('{}\n{}'.format(dialogue, _list))
            _dialogue  = (self.SD+self.DC).join(_list) + self.SD
            _dialogue = self.SD + DS + _dialogue
            # print(_dialogue)

            txt = txt.replace(dialogue, _dialogue, )
        return txt

    def split_sentece(self, txt):
        """ DELIMITER로 문장을 나누고, ''와 ' '는 제외함
        
            return: 문장들의 리스트
        """
        _list = [sen.strip() for sen in re.split(self.SD, txt)] 
        _list = list(filter(('').__ne__, _list))
        return _list

    
    def pos_tagging_chapter(self, sen_list):
        chap_tagged = []
        for i, line in enumerate(sen_list):
            tmp_line = []
            for tu in self.tagger.pos(line):
                if tu[0] in [self.DS1, self.DS2, self.DC]:
                    tmp_line.append(tu[0])
                else:
                    tmp_line.append('/'.join(tu))

                if tu[1] == 'UNKNOWN':
                    self.logger.debug('UNKNOWN TAG: {} <- {}'.format('/'.join(tu), line))
                    self.unknowns['/'.join(tu)] += 1

            if tmp_line not in [self.DS1, self.DS2, self.DC]:
                chap_tagged.append(' '.join(tmp_line))
        return chap_tagged

    def pos_tagging_book(self, cleaned):
        tagged = []
        for chap in cleaned:
            tagged.append(self.pos_tagging_chapter(chap))
        return tagged

        

