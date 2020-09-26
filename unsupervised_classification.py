from sklearn.cluster import KMeans
import multiprocessing
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string as string_val
import random
import tensorflow as tf


stop = stopwords.words('english')
MAIN_PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\TextClassification\\project_data\\'
type_voc = '_headlines'
# type_voc = ''
PATH = MAIN_PATH + 'preprocessed_data_unsupervised{}.txt'.format(type_voc)

class WordsTwoVec:
    def __init__(self, df):
        self.sent = df.tolist()
        self.phrases = Phrases(self.sent, min_count=1, threshold=1)
        self.bigram = Phraser(self.phrases)
        self.sentences = self.bigram[self.sent]
        self.w2v_model = Word2Vec(min_count=3, window=4, size=300, sample=1e-5, alpha=0.03,
                                  min_alpha=0.0007, negative=20, workers=multiprocessing.cpu_count()-1)
    def train(self):
        self.w2v_model.build_vocab(self.sentences)
        self.w2v_model.train(self.sentences, total_examples=self.w2v_model.corpus_count, epochs=30, report_delay=1)
        self.w2v_model.init_sims(replace=True)

    def save(self, name):
            self.w2v_model.save('models\\{}'.format(name))

def replace_commons(string):
    reformat = ['us ', 'ag ', 'china ', 'japan ', 'euro ', 'eu ', 'hong kong', 'uk ', 'united ', 'york ',
                'european ', 'europe ', 'india ', 'world ', 'florida ', 'africa ', 'asia ', 'german ', 'asian ']
    for r in reformat:
        res = ''.join(random.choices(string_val.ascii_lowercase, k=5))
        string = string.replace(r, res + ' ')
    return string

def str_split(string):
    split_str = string.split()
    pct = int(len(split_str) * 0.7)
    res = []
    [res.append(x) for x in split_str if x not in res]
    return ' '.join(res[:pct])

def replace_sentiment_words(word, sentiment_dict):
    '''
    replacing each word with its associated sentiment score from sentiment dict
    '''
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out

def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score

    inspired  by function from this wonderful article:
    https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34

    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer

    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    # if x.values[0] == 'china biggest steelmaker said its third quarter net income rose percent and reversed three straight quarters of declines as demand recovered':
    #     print('stop')

    dictionary = create_tfidf_dictionary(x, transformed_file, features)
    try:
        result = list(map(lambda y: dictionary[f'{y}'], x['Articles'].split()))
    except KeyError as e:
        print(e)
        result = e

    return result

def get_corpus(PATH):
    raw_articles = pd.read_table(PATH, header=None, names=['Articles'], skip_blank_lines=True)
    raw_articles = raw_articles[raw_articles['Articles'].str.contains('to contact the ') == False]
    raw_articles = raw_articles[raw_articles['Articles'].str.contains(' compiled by bloomberg ') == False]
    raw_articles['Articles'] = raw_articles['Articles'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    raw_articles['Articles'] = raw_articles['Articles'].str.replace('.', '')
    uncured_articles = raw_articles
    uncured_articles['Lenght Match'] = uncured_articles['Articles'].apply(lambda x: list(map(len, x.split())))
    uncured_articles['Lenght Match'] = uncured_articles['Lenght Match'].apply(lambda x: True if len([r for r in x if r > 4]) > 4 else False)
    uncured_articles = uncured_articles[uncured_articles['Lenght Match'] == True]
    del uncured_articles['Lenght Match']
    # uncured_articles['Articles'] = uncured_articles['Articles'].apply(str_split)
    count = uncured_articles['Articles'].str.split().str.len()
    uncured_articles = uncured_articles[~(count < 10)].reset_index(drop=True)
    uncured_articles['Articles'] = uncured_articles['Articles'].apply(replace_commons)
    uncured_articles.reset_index(drop=True, inplace=True)
    model_corpus = uncured_articles['Articles'].apply(lambda x: x.split())
    return model_corpus, uncured_articles

def build_model(model_corpus):
    init = WordsTwoVec(model_corpus)
    init.train()
    init.save('unsupervised_word2vec{}.model'.format(type_voc))
    word_vectors = init.w2v_model.wv
    del init.w2v_model # save memory
    model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=100).fit(X=word_vectors.vectors)
    words = pd.DataFrame(word_vectors.vocab.keys())
    words.columns = ['words']
    words['vectors'] = words['words'].apply(lambda x: word_vectors.wv[f'{x}'])
    words['cluster'] = words['vectors'].apply(lambda x: model.predict([np.array(x)]))
    words.cluster = words['cluster'].apply(lambda x: x[0])
    words['cluster_value'] = [1 if i == 0 else -1 for i in words.cluster]
    words['closeness_score'] = words.apply(lambda x: 1 / (model.transform([x['vectors']]).min()), axis=1)
    words['sentiment_coeff'] = words['closeness_score'] * words['cluster_value']
    words.to_csv('metrics_results\\predictive_scores.csv', index=False)
    return words

def get_predictions(sentiment_map, uncured_articles):
    sentiment_dict = dict(zip(sentiment_map['words'].values, sentiment_map['sentiment_coeff'].values))
    file_weighting = uncured_articles.copy()
    tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
    tfidf.fit(file_weighting['Articles'])
    features = pd.Series(tfidf.get_feature_names())
    transformed = tfidf.transform(file_weighting['Articles'])
    replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
    replaced_closeness_scores = file_weighting['Articles'].apply(
        lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
    replacement_df = pd.DataFrame(data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting['Articles']]).T
    replacement_df['rate'] = np.nan
    replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence', 'sentiment']
    replacement_df['sentiment_rate'] = replacement_df.apply(
        lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
    replacement_df['prediction'] = (replacement_df['sentiment_rate'] > 0).astype('int8')
    replacement_df.to_csv('metrics_results\\results{}.csv'.format(type_voc), index=False)
    return replacement_df


if __name__=='__main__':
    BUILD = False
    model_corpus, uncured_articles = get_corpus(PATH)
    if BUILD:
        words = build_model(model_corpus)
        words.to_csv('metrics_results\\sentiment_dictionary{}.csv'.format(type_voc), index=False)
    else:
        words = pd.read_csv('metrics_results\\sentiment_dictionary{}.csv'.format(type_voc))
        # word_vectors = Word2Vec.load('models\\unsupervised_word2vec{}.model'.format(type_voc)).wv
        # word_vectors.save_word2vec_format("vect.txt", binary=False)

    sentiment_map = words[['words', 'sentiment_coeff']]
    predictions = get_predictions(sentiment_map, uncured_articles)


