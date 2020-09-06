from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import multiprocessing
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from gensim.models.phrases import Phrases, Phraser
import pandas as pd
import numpy as np

MAIN_PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\TextClassification\\project_data\\'
PATH = MAIN_PATH + 'preprocessed_data_unsupervised.txt'  #'financial_articles_special_char.txt'


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

class WordsTwoVec:
    def __init__(self, documnet, df):
        # self.sentences = Text8Corpus(datapath(documnet))
        self.sentences = df['Articles'].tolist()
        self.phrases = Phrases([self.sentences], min_count=1, threshold=1)  #try with the list check if you need it on bottom too
        self.bigram = Phraser(self.phrases)
        sent = [row for row in df['Articles']]
        self.sentences = self.bigram[sent]
        # sentences[1]
        # self.docs = documnet
        self.w2v_model = Word2Vec(min_count=3,
                             window=4,
                             size=300,
                             sample=1e-5,
                             alpha=0.03,
                             min_alpha=0.0007,
                             negative=20,
                             workers=multiprocessing.cpu_count() - 1)

    def cleaning(sefl, doc):
        # Lemmatizes and removes stopwords
        # doc needs to be a spacy Doc object
        txt = [token.lemma_ for token in doc if not token.is_stop]
        # Word2Vec uses context words to learn the vector representation of a target word,
        # if a sentence is only one or two words long,
        # the benefit for the training is very small
        if len(txt) > 2:
            return ' '.join(txt)

if __name__=='__main__':
    # nlp = spacy.load('en', disable=['ner', 'parser'])  # disabling Named Entity Recognition for speed
    #     # brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])
    #     # t = time()
    #     # txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
    #     # print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    #     # df_clean = pd.DataFrame({'clean': txt})
    #     # df_clean = df_clean.dropna().drop_duplicates()
    BUILD = False
    raw_articles = pd.read_table(PATH, header=None, names=['Articles'], skip_blank_lines=True)
    raw_articles = raw_articles[raw_articles['Articles'].str.contains('to contact the ') == False]
    raw_articles = raw_articles[raw_articles['Articles'].str.contains(' compiled by bloomberg ') == False]
    count = raw_articles['Articles'].str.split().str.len()
    uncured_articles = raw_articles[~(count < 6)].reset_index(drop=True)

    if BUILD:
        init = WordsTwoVec(PATH, uncured_articles)
        init.w2v_model.build_vocab([init.sentences], progress_per=50000)
        init.w2v_model.train(init.sentences, total_examples=init.w2v_model.corpus_count, epochs=30, report_delay=1)
        init.w2v_model.init_sims(replace=True)
        init.w2v_model.save('unsupervised_word2vec.model')

        word_vectors = Word2Vec.load('unsupervised_word2vec.model').wv
        model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=word_vectors.vectors)
        word_vectors.similar_by_vector(model.cluster_centers_[0], topn=10, restrict_vocab=None)
        positive_cluster_center = model.cluster_centers_[0]
        negative_cluster_center = model.cluster_centers_[1]
        words = pd.DataFrame(word_vectors.vocab.keys())
        words.columns = ['words']
        words['vectors'] = words.words.apply(lambda x: word_vectors.wv[f'{x}'])
        words['cluster'] = words.vectors.apply(lambda x: model.predict([np.array(x)]))
        words.cluster = words.cluster.apply(lambda x: x[0])
        words['cluster_value'] = [1 if i == 0 else -1 for i in words.cluster]
        words['closeness_score'] = words.apply(lambda x: 1 / (model.transform([x.vectors]).min()), axis=1)
        words['sentiment_coeff'] = words.closeness_score * words.cluster_value
        print(words.head(10))
        words[['words', 'sentiment_coeff']].to_csv('sentiment_dictionary.csv', index=False)
        sentiment_map = words[['words', 'sentiment_coeff']]
    else:
        sentiment_map = pd.read_csv('kmeans\\sentiment_dictionary.csv')

    sentiment_dict = dict(zip(sentiment_map['words'].values, sentiment_map['sentiment_coeff'].values))
    file_weighting = uncured_articles.copy()
    tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
    tfidf.fit(file_weighting['Articles'])
    features = pd.Series(tfidf.get_feature_names())
    transformed = tfidf.transform(file_weighting['Articles'])
    replaced_tfidf_scores = file_weighting.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
    replaced_closeness_scores = file_weighting['Articles'].apply(
        lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
    replacement_df = pd.DataFrame(
        data=[replaced_closeness_scores, replaced_tfidf_scores, file_weighting['Articles']]).T  # ,file_weighting.rate
    replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence', 'sentiment']
    replacement_df['sentiment_rate'] = replacement_df.apply(
        lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
    replacement_df['prediction'] = (replacement_df['sentiment_rate'] > 0).astype('int8')
    replacement_df['sentiment'] = [1 if i == 1 else 0 for i in replacement_df['sentiment']]

    predicted_classes = replacement_df.prediction
    y_test = replacement_df.sentiment

    conf_matrix = pd.DataFrame(confusion_matrix(replacement_df['sentiment'], replacement_df['prediction']))
    print('Confusion Matrix')
    print(conf_matrix)

    test_scores = accuracy_score(y_test, predicted_classes), precision_score(y_test, predicted_classes), recall_score(
        y_test, predicted_classes), f1_score(y_test, predicted_classes)

    print('\n \n Scores')
    scores = pd.DataFrame(data=[test_scores])
    scores.columns = ['accuracy', 'precision', 'recall', 'f1']
    scores = scores.T
    scores.columns = ['scores']
    print(scores)