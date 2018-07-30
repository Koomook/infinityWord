""" Doesn't Work """

import re
import os, sys
from konlpy.tag import Mecab
from collections import Counter
import numpy as np
import logging

class Converter(object):
    """Convert text from raw novel data to structured text datal.
    Converting includes (1) Seperating into Chapters and line and Cleaning them,
    (2) Tagging POS of words line by line (3) Split long line into multiple liens.
    Standard format of outputs from methods should be: `Book > Chapters > Lines`

    Attributes:
        txt_type: String, it should be one of the 'N', 'D', 'E'
        mono: Boolean, if `True`, distinguish monologue and dialogue by two tokens
        logging_name: name of logger
        logging_path: file path to save logs
    """

    def __init__(self, txt_type, mono=True, logging_name='textconvert', logging_path=None):
        self.mono = mono
        self.tagger = Mecab()
        self.unknowns = Counter()

        self.logger = logging.getLogger(logging_name)

        file_handler = logging.FileHandler(filename=logging_path, mode='w', encoding='utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s', datefmt='%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)
        # Tokens
        self.SD = '🐱'  # 문장 구분 (Sentence Delimeter)
        self.D_SYMBOLS = ['"', "'", '‘', '’', '“', '”']
        self.DS1 = '⎡'  # 대사의 시작 문장 (Dialogue Start), Double quote
        self.DS2 = '⎛'  # 대사의 시작 문장 (Dialogue Start), Single quote
        self.DC = '⎜'  # 대사의 이어지는 문장 (Dialogue Continue)
        self.LS = '⌙'  # tagged data의 긴 나눈 표시 (Long Split)

        # Patterns
        if txt_type == 'N':
            self.P_CHAPTER = '<title>.*\n'
            self.P_WHITESPACE = '\r|\t|\\u200a|\\uf000|\\ufeff|\\u3000|\\xa0'
            self.P_DELIMITER = '(?<!\d)\.(?![\d]+)|\!|\?|\n'  # \n 포함
            self.P_DIALOGUE = '<talk>.*\n|\n *".+?"|\n *\'.+?\'|\n *‘.+?’|\n *“.+?”|\n *「.+?」'

        elif txt_type == 'D':  # \n 을 먼저 제거하는 과정이 꼭 필요함
            self.P_CHAPTER = '\n[제 ]*[\d]{1,3}[ ]*[장편막부]+ .{0,50}\n|\n[ ]*[\d]{1,3}[\. ]+.{0,50}\n|\n\n[가-힣]+\n\n|\n[ ]*[\d]{1,3}[ ]*\n|@ff|={10,}|-{10,}'
            self.P_WHITESPACE = '\n|\r|\t|\\u200a|\\uf000|\\ufeff|\\u3000|\\xa0'
            self.P_DELIMITER = '(?<!\d)\.(?![\d]+)|\!|\?'
            self.P_DIALOGUE = '".+?"|\'.+?\'|‘.+?’|“.+?”|「.+?」|\[.+\]'

        elif txt_type == 'E':
            self.P_CHAPTER = '\n[제 ]*[\d]{1,3}[ ]*[장편막부]+ .{0,35}\n|\n[ ]*[\d]{1,3}[\. ]+.{0,35}\n|\n\n[가-힣]+\n\n|\n[ ]*[\d]{1,3}[ ]*\n|@ff'
            self.P_WHITESPACE = '\r|\t|\\u200a|\\uf000|\\ufeff|\\u3000|\\xa0'
            self.P_DELIMITER = '(?<!\d)\.(?![\d]+)|\!|\?|\n'
            self.P_DIALOGUE = '\n *".+?"|\n *\'.+?\'|\n *‘.+?’|\n *“.+?”|\n *「.+?」'

        else:
            self.logger.warning(
                '{} is not defined text type. You only be able to tag and split sentences.'.format(txt_type))

        self.P_PARANTHESIS = '\(.+?\)|\[\d+?\]|\d+\)|〔.+?\〕'

        # Permitted symbols (Special Characters)
        self.P_PRMT_SYMBOLS = '%&~…\.\?\!,\+\-{}{}{}{}'.format(self.SD, self.DS1, self.DS2, self.DC)
        # Specific cases where permitted symbols should be skipped and usual special symbols
        self.P_SKIP_SYMBOLS = '<talk>|(?<!\d)[~\+\-]+(?![\d]+)|(?<!\d)%|[/\'\"\*\^\(\)\|‘’“”`;·:〈〉<>「」『』《》【】〖〗─=―—ㅡ_⎯]'

        # Filters
        self.F_KorNum = '[^ 가-힣0-9\\u1100-\\u11FF\\u3130-\\u318F{}]'.format(self.P_PRMT_SYMBOLS)
        self.F_KorEngNum = '[^ A-Za-z가-힣0-9\\u1100-\\u11FF\\u3130-\\u318F{}]'.format(self.P_PRMT_SYMBOLS)

        # Length Limits
        self.MIN_LEN_CHAPTER = 2500
        self.MIN_LEN_CHUNK = 5

    def clean_raw(self, book, clean_filter):
        """Seperate chapters by title of chapters, substitue or skip useless symbols,
        distinguish dialogue (and monologue) from statements, and split into sentneces.
        The order of methods is important.

        Attributes:
            book: String
            clean_filter: regular expression pattern to capture all unwanted characters.
                Although `skip_symbos method` works simillar, defining all unwnated characters
                one by one is impossible.
        """
        book_cleaned = []
        num_line_acc = 0
        for i, chap in enumerate(self.divide_chapter(book)):
            chap = self.skip_linespacing(chap)
            chap = self.skip_elaborate(chap)
            chap = self.sub_symbols(chap)
            chap = self.process_dialogue(chap)
            chap = self.skip_symbols(chap)
            sent_list = self.split_sentece(chap)

            _sent_list = []
            for sen in sent_list:
                if sen == self.DS1 or sen == self.DS2:
                    self.logger.debug('REVIEW CHAPTER {:3}: only dialogue start token'.format(i))
                else:
                    _not = re.findall(clean_filter, sen)
                    if _not != []:
                        if not _not in self.D_SYMBOLS:
                            self.logger.debug('-SKIPPED SYMBOLS: CHAPTER {:3}: {} <- {}'.format(i, _not, sen))
                        sent_only = re.sub(clean_filter, '', sen)
                        if not sent_only == '':
                            _sent_list.append(sent_only.strip())
                    else:
                        _sent_list.append(sen)
            len_list = len(_sent_list)
            num_line_acc += len_list
            self.logger.debug(
                '>> CHAPTER {:3}: {} characters, {} lines, {} lines accumulated'.format(i, len(chap), len_list,
                                                                                        num_line_acc))
            book_cleaned.append(_sent_list)
        self.logger.info('Cleaned: {} chapters: {} lines'.format(len(book_cleaned), num_line_acc))
        return book_cleaned

    def divide_chapter(self, txt):
        self.logger.debug('PART DELIMITERS: {}'.format(re.findall(self.P_CHAPTER, txt)))
        chapters = re.split(self.P_CHAPTER, txt)

        result = []
        for i, chapter in enumerate(chapters):
            self.logger.debug('PART {:3}: {}'.format(i, len(chapter)))
            if len(chapter) > self.MIN_LEN_CHAPTER:
                result.append(chapter)
            else:
                self.logger.debug('PART {:3} SKIPPED: {}'.format(i, chapter))
        return result

    def skip_linespacing(self, txt):  # 이름 바꾸기
        """Remove new lines to fit text and unnecessary margins """
        txt = re.sub(self.P_WHITESPACE, '', txt)
        return txt

    def skip_elaborate(self, txt):  # 부연, 주석표시 등 삭제
        """Remove additional explanation and annotation marks. """
        txt = re.sub(self.P_PARANTHESIS, '', txt)
        return txt

    def skip_symbols(self, txt):
        """Remove usual sepecial characters and emoji """
        txt = re.sub(self.P_SKIP_SYMBOLS, '', txt)
        return txt

    def sub_symbols(self, txt):
        """Subtitute symbols to make sure Sentence Delimeters and special marks
        (1) Shorten successive spaces
        (2) replace `...` to `…`, max length is two of `…`
        (3) Add SD after `.` except for the numbers(float)
        (4) Add SD after `!`, shorten the successivemarks to one
        (5) Add SD after `?`, shorten the successivemarks to one
        """

        txt = re.sub(' {2,}', ' ', txt)
        txt = re.sub('\.{2,3}', '…', txt)
        txt = re.sub('…{2,}', '……', txt)

        txt = re.sub('(?<!\d)\.(?![\d]+)', '.' + self.SD, txt)
        txt = re.sub('‼+|\!+', '!' + self.SD, txt)
        txt = re.sub('\?+', '?' + self.SD, txt)
        return txt

    def process_dialogue(self, txt):
        """Pick out monologues and dialogues and split to sentences
        and reunite them by length, because there are too short sentences
        in dialogues.
        """

        def _unite_short_chunks(start, _list, minimum):
            """If the length of a chunk is shorter than `minimum`,
            attach it to nearest chunk.
            """
            for i in range(start, len(_list) - 1):
                if (len(_list[i]) < minimum or len(_list[i + 1]) < minimum):
                    _list[i] = ''.join([_list[i], _list.pop(i + 1)])
                    _list = _unite_short_chunks(i, _list, minimum)
                    break
            return _list

        for dialogue in re.findall(self.P_DIALOGUE, txt):
            if self.mono == True and (dialogue.strip()[0] in ["'", "‘"]):
                DS = self.DS2
            else:
                DS = self.DS1

            _dialogue = self.skip_symbols(dialogue)
            _list = self.split_sentece(_dialogue)

            if len(_list) > 1:
                _list = _unite_short_chunks(0, _list, self.MIN_LEN_CHUNK)
            _dialogue = (self.SD + self.DC).join(_list) + self.SD
            _dialogue = self.SD + DS + _dialogue

            txt = txt.replace(dialogue, _dialogue, )
        return txt

    def split_sentece(self, txt):
        """Seprerate a chunk of text to list of strings
        by Sentence Delimeters
        """
        _list = [sen.strip() for sen in re.split(self.SD, txt)]
        _list = list(filter(('').__ne__, _list))
        return _list

    def pos_tagging_chapter(self, sent_list):
        """Tag list of sentneces with tagger (konlpy.tag.Mecab)
        unknown tags are collected in list.

        Attributes:
            sent_list: List of sentneces, a chapter.
        """
        chap_tagged = []
        for i, line in enumerate(sent_list):
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
        """Wrapper method for tagging chapters, a book """
        tagged = []
        for chap in cleaned:
            tagged.append(self.pos_tagging_chapter(chap))
        self.logger.info('Tagged: {} chapters'.format(len(tagged)))
        return tagged

    def split_long_sentence(self, sent_tagged, max_length, verbosa=False):
        """Split long sentence into two sentneces recursively.

        Attributes:
            sent_tagged: List of tagged sentences, a chapter
            max_length: criteria length to cut sentence
            verbosa: if True, all process of split is shown.
        """

        # split sentence into words list
        if verbosa: self.logger.debug('origin sentence: {}'.format(sent_tagged))
        word_arr = np.array(sent_tagged.split(' '))
        len_arr = len(word_arr)

        # split words and tags from tagged data to find specific pattern
        words, tags = [], []
        for word_tag in word_arr:
            try:
                if word_tag in ['⎡', '⎛', '⎜', '⌙']:
                    words.append(word_tag)
                    tags.append('TK')
                else:
                    word, tag = word_tag.split('/')
                    words.append(word)
                    tags.append(tag)
            except:
                self.logger.warning('{} is not permitted symbol: {}'.format(word_tag, sent_tagged))
                return [sent_tagged]

        if len(words) != len(tags):
            self.logger.info('lengths is not mathced between words and tags')

        # find indices of specific words
        split_indices = []
        for idx in range(1, len(word_arr) - 2):
            if (word_arr[idx] in [',/SC', '…/SE']):
                if tags[idx - 1] not in ['SN', 'NNG', 'NNP']:
                    split_indices.append(idx)
                else:
                    if tags[idx + 1][0] == 'M':
                        split_indices.append(idx)
            if (tags[idx] == 'EC') and (words[idx] in ['고', '지만', '으며', '으나', '으니']):
                if (tags[idx - 1].split('+')[-1] == 'EP') and (tags[idx + 1] != 'SC'):
                    split_indices.append(idx)
        split_indices = np.array(split_indices)

        # choose one index with some conditions
        if verbosa: self.logger.debug('split indices: {}'.format(split_indices))

        if not len(split_indices) == 0:
            down = int(len_arr * (1 / 4))
            up = int(len_arr * (3 / 4))
            in_middle = split_indices[(split_indices > down) & (split_indices < up)]
            if verbosa: self.logger.debug('split indices in middle: {}'.format(in_middle))

            if not len(in_middle) == 0:
                if len(in_middle) == 1:
                    split_idx = in_middle[0]
                else:
                    split_idx = in_middle[(np.abs(in_middle - len_arr / 2)).argmin()]
                if verbosa: self.logger.debug('split_idx: {}'.format(split_idx))

            else:
                if verbosa: self.logger.debug('no spliter in middle')
                return [sent_tagged]
        else:
            if verbosa: self.logger.debug('no_spliter')
            return [sent_tagged]

        # split list
        list1 = word_arr[:split_idx + 1]
        list2 = np.insert(word_arr[split_idx + 1:], 0, self.LS, axis=0)

        sent1 = ' '.join(list1)
        sent2 = ' '.join(list2)
        if verbosa: self.logger.debug('splited sentence 1: {}'.format(sent1))
        if verbosa: self.logger.debug('splited sentence 2: {}'.format(sent2))

        result = []

        # check length of sublist and execute this method recusively

        if len(list1) > max_length:
            result.extend(self.split_long_sentence(sent1, max_length))
        else:
            result.append(sent1)

        if len(list2) > max_length:
            result.extend(self.split_long_sentence(sent2, max_length))
        else:
            result.append(sent2)

        return result

    def split_long_sentence_book(self, tagged_book, max_length, verbosa=False):
        """Wrapper method for split long sentences in chapters, a book
        If any sub-sentnece is longer than max_length and not split,
        cancle all split action taken and return the origin sentence
        """

        split_book = []
        num_split, num_not, num_total = 0, 0, 0

        for chap in tagged_book:
            split_chap = []
            for sent in chap:
                num_total += 1
                sent = sent.strip(' ')
                ln = len(sent.split(' '))

                if ln > max_length:
                    split = self.split_long_sentence(sent, max_length, verbosa)
                    if len(split) > 1:
                        len_split = np.array([len(s.split(' ')) for s in split])
                        if len(np.where(len_split > max_length)[0]) == 0:
                            num_split += 1
                            split_chap.extend(split)
                        else:  # Split but one of sub-sentence is still long
                            split_chap.append(sent)
                            num_not += 1
                    else:  # Not Split
                        split_chap.append(sent)
                        num_not += 1
                else:  # original sentence is shorter than max_length
                    split_chap.append(sent)
            split_book.append(split_chap)
        self.logger.info('Split: total {:7} line | Split: {} | Not: {}'.format(num_total, num_split, num_not))
        return split_book
