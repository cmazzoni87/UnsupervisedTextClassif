import re
import glob
import os
from multiprocessing import Pool
import datetime
__author__= 'Claudio Mazzoi'


STRIP_SPECIAL_CHARS = re.compile("[^A-Za-z0-9 ]+")
INVALID_WORDS = 'https|http|www|.com|inc.|co.|corp.|llc|.html|.net|ltd|.gov'
DATASET_PATH = 'D:\\\\Datasets\\financial-news-dataset\\financial-news-dataset-raw\\'
BLOOMBERG_ART = glob.glob(DATASET_PATH + 'bloomberg_news\\*', recursive=False)
REUTERS_ART = glob.glob(DATASET_PATH + 'reuters_news\\*', recursive=False)

def reader(file):
    suffixes = {' u k ': ' uk ', ' u s ': ' us ', ' u e ': ' ue ',
                ' s ': ' ', ' we ve ': ' we ', ' t z bloomber news ': ' '}
    # negative_stops = ["couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    #                   "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    #                   "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    #                   "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    encoding = "mbcs"   #'utf-8'
    with open('financial_articles.txt', 'a') as the_file:
        with open(file, "r", encoding=encoding) as f:
            article = f.read()[4:]
            article = article.replace('\n', ' ')
            if '========================' not in article:
                if 'reuters_news' in file:
                    article = re.sub('^.*?(Reuters)', "", article) # remove all string before reaches reuters
                if 'bloomberg_news' in file:
                    article = re.sub('^.*?.html', "", article) # remove all string before reaches reuters
                    article = re.sub('To contact the reporter on this story.*', "", article) # remove all string before reaches reuters
                    article = re.sub('To contact the reporters on this story.*', "", article) # remove all string before reaches reuters

                article = article.replace('.', '')
                # article = re.sub(r'\b(.+)\s+\1\b', r'\1', article) # remove duplicate words, slows code significantly
                article = re.sub(r'\([^)]*\)', ' ', article ) # removes words inside parentheses
                article = article.lower()
                article = re.sub(INVALID_WORDS, ' ', article)
                article = re.sub(STRIP_SPECIAL_CHARS, ' ', article) #remove special char
                for key, val in suffixes.items():
                    article = article.replace(key, val)
                article = re.sub('\d', '9', article)
                # article = re.sub(r'\b[a-zA-Z]\b', ' ', article) # remove single character words
                article = re.sub(' +', ' ', article) # remove extra spaces
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

