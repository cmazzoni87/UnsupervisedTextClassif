import re
import glob
import os
from multiprocessing import Pool
import datetime
__author__= 'Claudio Mazzoi'


STRIP_SPECIAL_CHARS = re.compile("[^A-Za-z ]+")
DATASET_PATH = 'D:\\\\Datasets\\financial-news-dataset\\financial-news-dataset-raw\\'
BLOOMBERG_ART = glob.glob(DATASET_PATH + 'bloomberg_news\\*', recursive=False)
REUTERS_ART = glob.glob(DATASET_PATH + 'reuters_news\\*', recursive=False)


def reader(file):
    encoding = "mbcs"   #'utf-8'
    with open('financial_articles.txt', 'a') as the_file:
        with open(file, "r", encoding=encoding) as f:
            article = f.read()[7:]
            article = article.replace('\n', ' ')
            if '========================' not in article:
                if 'reuters_news' in file:
                    article = re.sub('^(.*?Reuters )', "", article) # remove all string before reaches reuters
                # article = re.sub(r'\b(.+)\s+\1\b', r'\1', article) # remove duplicate words, slows code significantly
                article = re.sub(r'\b[a-zA-Z]\b', ' ', article) # remove single character words
                article = re.sub(r'\([^)]*\)', ' ', article) # removes words inside parentheses
                article = re.sub(STRIP_SPECIAL_CHARS, ' ', article) #remove special char
                article = article.lower()
                article = article.replace('https', ' ')
                article = article.replace('www', ' ')
                article = article.replace('http', ' ')
                article = article.replace('.com', ' ')
                article = article.replace('inc.', ' ')
                article = article.replace('co.', ' ')
                article = article.replace('corp.', ' ')
                article = article.replace('llc', ' ')
                article = article.replace('.html', ' ')
                article = article.replace('.net', ' ')
                article = article.replace(' . ', ' ')
                article = article.replace(' , ', ' ')
                article = article.replace(' u k ', ' uk ')
                article = article.replace(' u s ', ' us ')
                article = article.replace(' u e ', ' ue ')
                article = article.replace(' s ', ' ')
                article = re.sub(' +', ' ', article) # remove extra spaces
                article = article.strip()
                the_file.write(article + '\n')
        f.close()
    the_file.close()

if __name__ == '__main__':
    """
    File cleaner and parser using multiprocessing
    """
    files = []
    CORES = 4  # number of cores to use
    for p in BLOOMBERG_ART:
        files.extend(glob.glob(p + '\\*'))
    for q in REUTERS_ART:
        files.extend(glob.glob(q + '\\*'))
    files = [file for file in files if os.stat(file).st_size > 0]
    print('Process Started at:')
    print(datetime.datetime.now())
    pool = Pool(CORES)
    pool.map(reader, files)
    print('Process Ended at')
    print(datetime.datetime.now())

