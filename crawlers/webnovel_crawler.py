""" Doesn't Work """

import requests
import re
import os
import time
import codecs
from bs4 import BeautifulSoup
from os.path import dirname, abspath, join, exists

# 오늘의 웹소설 : http://novel.naver.com/webnovel/weekdayList.nhn

WEBNOVEL_BASE_URL = 'http://novel.naver.com'
GENRE_BASE_URL = 'http://novel.naver.com/webnovel/genre.nhn?'

TITLE_TOKEN = '<title>'
TALK_TOKEN = '<talk>'
QUOTES = '[“”‘’-]'  # will be stripped

# 로맨스, 로맨스판타지, 판타지, 무협,미스터리,역사/전쟁, 라이트노벨, 퓨전
# 장르 이름 : [url_num, 'url_sep']
GENRE_ITEMS = {
    'romance' : [101, 'rom'],
    'romancefantasy' : [109, 'rof'],
    'fantasy' : [102, 'sff'],
    'action' :[103, 'hro'],
    'mystery' : [104,'mth'],
    'history&war' :[105, 'his'],
    'lightnovel' :[106, 'lno'],
    'fusion' : [108, 'fus']
}


# 장르별 소설 url dict 가져옴
def get_novel_url_dict(genre_name, finished):
    novel_dict = {}  # 소설 제목 : 소설 url 형태로 return

    genre_num, url_sep = GENRE_ITEMS[genre_name]

    for page_num in range(1, 20):
        params = {
            'genre': genre_num,
            'page': page_num,
        }

        if finished:  # 완결작일경우 url 다름
            params['order'] = 'Read'
            params['finish'] = 'true'

        r = requests.get('http://novel.naver.com/webnovel/genre.nhn?', params=params)
        crawler = BeautifulSoup(r.content, 'html.parser')

        class_temp = 'list_type1 v2 NE=a:lst_' + url_sep
        novel_list = crawler.find('ul', {'class': class_temp}).findAll('li')
        for novel in novel_list:
            url = novel.a.attrs['href']  # url
            title = novel.find('p', {'class': 'subj v3'}).contents[0]  # title
            if title in novel_dict:
                print(str(page_num) + '페이지에서 완료')
                return novel_dict
            else:
                novel_dict[title] = WEBNOVEL_BASE_URL + url
        time.sleep(0.05)
    return novel_dict


# 유효한 response인지 검사
def valid_status(response):
    if response.status_code == 200:
        return True
    else :
        return False


# url로부터 내용 읽어서 파싱
def get_content_from_url(content_page_url):
    content = ''
    r = requests.get(content_page_url)
    if valid_status(r):
        pass
    else:
        print("오류 발생 : {}".format(content_page_url))
        return ''

    crawler = BeautifulSoup(r.content, 'html.parser')

    title = crawler.find('h2', {'class': 'detail_view_header'}).contents[0]  # 몇화 제목
    content += TITLE_TOKEN + title + '\r\n\r\n'

    line_list = crawler.find('div', {'class': 'detail_view_content ft15'}).find_all('p')
    for line in line_list:
        if not line.has_attr('class'):  # general sentence
            processed_sentence = line.get_text().strip()
            content += processed_sentence
        elif 'talk' in line['class']:  # talk
            processed_talk = line.get_text().strip().strip(QUOTES).strip()
            content += TALK_TOKEN + processed_talk
        elif 'pic' in line['class']:  # picture
            pass
        else:
            print('기타 태그 오류 발생')
            print(line)
        content = content + '\r\n'
    time.sleep(0.05)

    return content


# url_list로부터 읽어와서 텍스트 파일에 저장
def crawl_and_write(genre, novel_url_dict, finished):
    if finished:
        finish_string = '완결'
    else:
        finish_string = '미완결'
    print('{} {} 장르 총 url {}개'.format(finish_string, genre, len(novel_url_dict)))
    print('----------------------------------------')
    check_index = 0

    for title, url in novel_url_dict.items():
        check_index = check_index + 1
        file_name = title + '.txt'  # filename
        file_name = re.sub('[/:*?"<>|]', '', file_name)  # file명 형식에 맞게 수정

        r = requests.get(url)
        crawler = BeautifulSoup(r.content, 'html.parser')

        total = int(crawler.find("span", {"class": "total"}).get_text()[1:-1])  # 전체 회차
        if total == 0:  # 회차 없으면 멈춤
            break
        page_limit = total // 10  # 한 페이지에 10개씩
        if total % 10 != 0:
            page_limit += 1
        #         print(total)
        #         print(page_limit)

        iterate_range = range(1, page_limit + 1)

        novel_content_list = []
        for page_num in iterate_range:
            r = requests.get(url, params={'page': page_num})
            #         print(r.url)
            crawler = BeautifulSoup(r.content, "html.parser")
            page_novel_content_list = crawler.find("ul", {"class": "list_type2 v3 NE=a:lst"}).findAll(
                "li")  # 이 페이지의 모든 소설 리스트
            novel_content_list.extend(page_novel_content_list)

        if not finished:
            novel_content_list.reverse()

        write_string = ''
        for novel_content in novel_content_list:
            content_url = novel_content.a.attrs['href']
            content_page_url = WEBNOVEL_BASE_URL + content_url  # 소설 내용 url
            write_string = write_string + get_content_from_url(content_page_url)

        with codecs.open(os.path.join(directory, file_name), 'w', encoding='utf-8') as f:  # file에 write
            f.write(write_string)
        print('{} : 저장 성공'.format(title))

    # 진행 상황 출력
    if check_index % 10 == 0:
        print("** 현재 " + str(check_index) + "개 진행중...")


# metadata
# {'10도, 우리 연애의 온도차': ['설래인', 483047, 8, 9.97, 24066]}
# {제목 : [작가,novelid, 총 회차, 평점, 관심수]}
def get_metadata_dict(genre_name, finished):
    metadata_dict = {}

    genre_item = GENRE_ITEMS[genre_name]
    genre_num = genre_item[0]
    url_sep = genre_item[1]

    for page_num in range(1, 20):
        params = {
            'genre': genre_num,
            'page': page_num,
        }

        if finished:  # 완결작일경우 url 다름
            params['order'] = 'Read'
            params['finish'] = 'true'

        r = requests.get('http://novel.naver.com/webnovel/genre.nhn?', params=params)
        crawler = BeautifulSoup(r.content, 'html.parser')

        class_temp = 'list_type1 v2 NE=a:lst_' + url_sep
        novel_list = crawler.find('ul', {'class': class_temp}).findAll('li')
        for novel in novel_list:
            url = novel.a.attrs['href']  # url
            novel_id = url.split('=')[-1]  # novel_id
            title = novel.find('p', {'class': 'subj v3'}).contents[0]  # title
            title = re.sub('[/:*?"<>|]', '', title)  # 특수문자 제거
            author = novel.find('span', {'class': 'ellipsis'}).get_text()  # author
            total_episode = novel.find('span', {'class': 'num_total'}).get_text()[2:-1]  # total_episode
            star_rating = novel.find('span', {'class': 'score_area'}).get_text()[2:]  # 별점
            attention_rating = novel.find('span', {'class': 'info_text'}).get_text()[2:].replace(',', '')  # 관심 개수
            if attention_rating[-1] == '만':
                attention_rating = float(attention_rating[:-1]) * 10000

            if title in metadata_dict:
                print(str(page_num) + '페이지에서 완료')
                return metadata_dict
            else:
                metadata_dict[title] = [author, int(novel_id), int(total_episode), float(star_rating),
                                        int(attention_rating)]
        time.sleep(0.05)
    return metadata_dict


if __name__ == '__main__':
    base_dir = dirname(dirname(abspath(__file__)))
    data_dir = join(base_dir, 'data', 'webnovel_data')
    for genre in GENRE_ITEMS.keys():
        directory = join(data_dir, genre)
        if not exists(directory):
            os.makedirs(directory)

        for finished in (True, False):  # 완결, 미완결
            if (genre == 'history&war') & (finished == True):
                continue
            novel_url_dict = get_novel_url_dict(genre, finished)  # 장르별 소설 url list
            crawl_and_write(genre, novel_url_dict, finished)  # 파일에 저장
            print('\n')
        print('\n')

    metadata_dict = {}
    for genre in GENRE_ITEMS.keys():
        for finished in (True, False):  # 완결, 미완결
            if (genre == 'history&war') & (finished == True):
                continue
            metadata_dict.update(get_metadata_dict(genre, finished))
        print('** %s 장르 완료' % genre)

    import pickle

    # Save metadata as pickle file
    with open('metadata_dict.pickle', 'wb') as handle:
        pickle.dump(metadata_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)