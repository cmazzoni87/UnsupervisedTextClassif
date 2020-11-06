import re
import glob
import os
from multiprocessing import Pool
import datetime
from stop_words import *
# from trie import Trie
__author__= 'Claudio Mazzoni'

stop = STOP_WORDS_LARGE
# REMOVE_STOP = "{}".format(' | '.join(stop))
# t = Trie()
# for w in st op.sort():
#     t.add(w)
# REMOVE_STOP = t.pattern()
# STRIP_SPECIAL_CHARS = re.compile("[a-zA-Z0-9,.!? ]")     #"[^A-Za-z0-9' ]+") #!?$
# INVALID_WORDS = 'https|http|www.|.com|inc.|co.|corp.|llc.|.html|.net|ltd.|ltd|.gov'
DATASET_PATH = 'D:\\\\Datasets\\financial-news-dataset\\financial-news-dataset-raw\\'
BLOOMBERG_ART = glob.glob(DATASET_PATH + 'bloomberg_news\\*', recursive=False)
REUTERS_ART = glob.glob(DATASET_PATH + 'reuters_news\\*', recursive=False)

def reader(params, headline=False):
    file = params
    file_name = 'preprocessed_data_unsupervised_short'
    if headline:
        file_name = 'preprocessed_data_unsupervised_headlines'
    suffixes = {' u k ': ' unitedkindom ', ' u s ': ' unitedstates ', ' e u ': ' eurounion ', ' s ': ' ',
                ' we ve ': ' we ', ' t z bloomber news ': ' '}
    encoding = "mbcs"
    with open('{}.txt'.format(file_name), 'a') as the_file:
        with open(file, "r", encoding=encoding) as f:
            if headline:
                if 'reuters' in file:
                    article = f.read().split('\n')[0]
                else:
                    article = file.split('\\')[-1].replace('-', ' ')
            else:
                article = f.read()[4:]
                article = article.replace('\n', ' ')
            article = re.sub(r'(?<!\w)([A-Z])\.', r'\1', article)
            article = article.lower()
            article = re.sub("\\b[A-Z]+(?:\\.|\\s+|$)", "", article)
            article = ''.join(article.split('html')[1:])
            article = article.replace("cia.", "")
            article = article.replace("u. s.", "united states")
            article = article.replace(".html ", " ")
            article = article.replace(".net ", " ")
            article = article.replace(" ltd.", " ")
            article = article.replace(" www.", " ")
            article = article.replace(".com ", " ")
            article = article.replace(" inc.", " ")
            article = article.replace(" co.", " ")
            article = article.replace(" corp.", " ")
            article = article.replace(" llc.", " ")
            article = article.replace(" u.s ", " united states ")
            article = article.replace(" u.s. ", " united states ")
            article = article.replace(" u.k ", " united kindom ")
            article = article.replace(" u.k. ", " united kindom ")
            article = article.replace(" e.u ", " european union ")
            article = article.replace(" e.u. ", " european union ")
            article = article.replace(" p.m ", " pm ")
            article = article.replace(" a.m ", " am ")
            article = article.replace(" p.m. ", " pm ")
            article = article.replace(" a.m. ", " am ")
            if len(article.split('.')) > 4:
                article = '. '.join(article.split('.')[:4])
            if len(article.split('.')) > 1:
                if '========================' not in article:
                    if 'reuters_news' in file:
                        article = re.sub('^.*?(Reuters)', "", article) # remove all string before reaches reuters
                    if 'bloomberg_news' in file:
                        article = re.sub('^.*?.html', "", article) # remove all string before reaches reuters
                        article = re.sub('To contact the reporter on this story.*', "", article) # remove all string before reaches reuters
                        article = re.sub('To contact the reporters on this story.*', "", article) # remove all string before reaches reuters
                        article = article.replace('according to data compiled by bloomberg', '')
                        article = article.replace('data compiled by bloomberg', '')
                    article = article.replace('â€™', '')
                    article = article.replace('A$', '')
                    article = article.replace('â€', '')
                    article = article.replace('â€œ', '')
                    article = re.sub(r'\([^)]*\)', ' ', article) # removes words inside parentheses
                    article = re.sub('following is a.*:', '', article)
                    article = re.sub('following is a.*\.', '', article)

                    article = re.sub('\..*\?', '. ', article)
                    article = re.sub('\..*!', '. ', article)
                    article = re.sub(' +', ' ', article)

                    article = re.sub("(\d)\.(\d)", '', article)
                    article = re.sub('\d', '', article)
                    article = article.replace(" https", " ")
                    article = article.replace(" http", " ")
                    article = article.replace(" ltd ", " ")
                    article = article.replace(".gov ", " ")
                    article = article.replace(" s ", " ")
                    article = re.sub('[^a-zA-Z0-9 ]', ' ', article) # remove special chars ^.,a-zA-Z0-9
                    article = article.replace(" ! ", "! ")
                    article = article.replace(" ? ", "? ")

                    article = article.replace(" ll ", "")
                    article = article.replace(" re ", "")
                    article = article.replace('sn t ', 'snt ')
                    article = article.replace(' i d ', ' id ')
                    for key, val in suffixes.items():
                        article = article.replace(key, val)
                    article = re.sub(' +', ' ', article) # remove extra spaces

                    article = '\n'.join([i.lstrip() for i in article.split('\n') if len(i.split(' ')) > 5])
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
    # files = [file for file in files
    #          if 'earning' in file.split('\\')[-1]
    #          or 'deal' in file.split('\\')[-1]
    #          or 'profit' in file.split('\\')[-1]
    #          or 'controversy' in file.split('\\')[-1]
    #          or 'stock' in file.split('\\')[-1]
    #          or 'sale' in file.split('\\')[-1]
    #          or 'aquire' in file.split('\\')[-1]
    #          or 'files' in file.split('\\')[-1]
    #          ]
    print('Process Started at:')
    print(datetime.datetime.now())
    pool = Pool(CORES)
    pool.map(reader, files)
    print('Process Ended at')
    print(datetime.datetime.now())

