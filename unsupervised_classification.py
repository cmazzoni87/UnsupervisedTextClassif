import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import multiprocessing
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import string as string_val
import random
import datetime
# import tensorflow as tf


stop = stopwords.words('english')
MAIN_PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\TextClassification\\project_data\\'
type_voc = '_headlines'  # '_short' #'
# type_voc = ''
PATH = MAIN_PATH + 'preprocessed_data_unsupervised{}.txt'.format(type_voc)
time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


class WordsTwoVec:
    def __init__(self, df):
        self.sent = df.tolist()
        self.phrases = Phrases(self.sent, min_count=30, threshold=1)
        self.bigram = Phraser(self.phrases)
        self.sentences = self.bigram[self.sent]
        self.w2v_model = Word2Vec(min_count=30, window=3, size=252, sample=6e-5, alpha=0.01,   # sample=1e-5
                                  min_alpha=0.0005, negative=5, workers=multiprocessing.cpu_count()-1)

    def train(self):
        self.w2v_model.build_vocab(self.sentences)
        self.w2v_model.train(self.sentences, total_examples=self.w2v_model.corpus_count, epochs=60, report_delay=1)
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
        out = 0 #this is an issue
    return out


def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo


def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    '''
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
    # sent_text = nltk.sent_tokenize('d')
    raw_articles['Articles'] = raw_articles['Articles'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
    raw_articles['Articles'] = raw_articles['Articles'].str.replace('.', '')
    uncured_articles = raw_articles
    uncured_articles['Lenght Match'] = uncured_articles['Articles'].apply(lambda x: list(map(len, x.split())))
    uncured_articles['Lenght Match'] = uncured_articles['Lenght Match'].apply(lambda x: True if len([r for r in x if r > 4]) > 4 and len([r for r in x if r > 4]) < 15 else False)
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
    del init.w2v_model  # save memory
    return word_vectors


def init_cluster(word_vectors):
    print(word_vectors.vectors)
    model = KMeans(n_clusters=3, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
    labels = model.labels_
    silhouette_score = metrics.silhouette_score(word_vectors.vectors, labels, metric='euclidean')
    print("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
    print(model.score(word_vectors.vectors))
    print("Silhouette_score: ")
    print(silhouette_score)
    print(word_vectors.similar_by_vector(model.cluster_centers_[0], topn=50, restrict_vocab=None))
    print(word_vectors.similar_by_vector(model.cluster_centers_[1], topn=50, restrict_vocab=None))
    print(word_vectors.similar_by_vector(model.cluster_centers_[2], topn=50, restrict_vocab=None))
    y_kmeans = model.predict(word_vectors.vectors)
    plt.scatter(word_vectors.vectors[:, 0], word_vectors.vectors[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
    words = pd.DataFrame(word_vectors.vocab.keys())
    words.columns = ['words']
    words = words[words['words'].str.len() > 3].reset_index(drop=True)
    words['vectors'] = words['words'].apply(lambda x: word_vectors.wv[f'{x}'])
    words['cluster'] = words['vectors'].apply(lambda x: model.predict([np.array(x)]))
    words['cluster'] = words['cluster'].apply(lambda x: x[0])
    words['cluster_value'] = [1 if i == 0 else -1 if i == 1 else 0 for i in words['cluster']]
    words['closeness_score'] = words.apply(lambda x: 1 / (model.transform([x['vectors']]).min()), axis=1)
    words['sentiment_coeff'] = words['closeness_score'] * words['cluster_value']
    words.to_csv('metrics_results\\predictive_scores_{}.csv'.format(time_stamp), index=False)
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
    replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
    # replacement_df['prediction'] = (replacement_df['sentiment_rate'] > 0).astype('int8')
    replacement_df.loc[(replacement_df['sentiment_rate'] >= 150), 'prediction'] = 1
    replacement_df.loc[(replacement_df['sentiment_rate'] >= -149) & (replacement_df['sentiment_rate'] <= 149), 'prediction'] = 0
    replacement_df.loc[(replacement_df['sentiment_rate'] <= -150), 'prediction'] = -1
    replacement_df.to_csv('metrics_results\\results{0}_{1}.csv'.format(type_voc, time_stamp), index=False)
    return replacement_df

def display_wordlist(model, wordlist):
    vectors = [model[word] for word in wordlist if word in model.wv.vocab.keys()]
    word_labels = [word for word in wordlist if word in model.wv.vocab.keys()]
    word_vec_zip = zip(word_labels, vectors)

    # Convert to a dict and then to a DataFrame
    word_vec_dict = dict(word_vec_zip)
    df = pd.DataFrame.from_dict(word_vec_dict, orient='index')

    # Use tsne to reduce to 2 dimensions
    tsne = TSNE(perplexity=65,n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(df)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display plot
    plt.figure(figsize=(16, 8))
    plt.plot(x_coords, y_coords, 'ro')

    for label, x, y in zip(df.index, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


if __name__ == '__main__':
    BUILD = False
    model_corpus, uncured_articles = get_corpus(PATH)
    sample_test = uncured_articles.sample(n=40, random_state=1)
    uncured_articles = uncured_articles.drop(sample_test.index)
    sample_test.reset_index(drop=True, inplace=True)
    uncured_articles.reset_index(drop=True, inplace=True)
    if BUILD:
        word_vectors = build_model(model_corpus)
        words = init_cluster(word_vectors)
        words.to_csv('metrics_results\\sentiment_dictionary{0}_{1}.csv'.format(type_voc, time_stamp), index=False)
    else:
        word_vectors = Word2Vec.load('models\\unsupervised_word2vec{}.model'.format(type_voc)).wv
        display_wordlist(word_vectors, list(word_vectors.vocab))
        # words = init_cluster(word_vectors)
    # sentiment_map = words[['words', 'sentiment_coeff']]
    # predictions = get_predictions(sentiment_map, sample_test)  # uncured_articles)
