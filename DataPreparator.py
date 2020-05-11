import pandas as pd
import nltk
from nltk.corpus import stopwords
import datetime
from dask import dataframe as dd

stop = stopwords.words('english')
MAIN_PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\TextClassification\\'
TOKENIZER = nltk.tokenize.WhitespaceTokenizer()
LEMMATIZER = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [LEMMATIZER.lemmatize(w) for w in TOKENIZER.tokenize(text)]

def wrapper_lemme(df):
    return df['Articles'].apply(lemmatize_text)

def data_converter(path):
    print('Process Started at:')
    print(datetime.datetime.now())
    uncured_articles = pd.read_table(path, header=None, names=['Articles'], skip_blank_lines=True)
    # uncured_articles['Articles'] = uncured_articles['Articles'].apply(lambda x: [item for item in x if item not in stop])
    uncured_articles['Tokenized Articles'] = uncured_articles['Articles'].apply(lemmatize_text)
    print('Process Ended at:')
    print(datetime.datetime.now())

    # uncured_articles = dd.from_pandas(uncured_articles, npartitions=2)
    # lemetized = uncured_articles.map_partitions(wrapper_lemme).compute(scheduler='processes')

    # gpu_uncured_articles = cudf.DataFrame.from_pandas(uncured_articles)
    # gpu_uncured_articles['Tokenized Articles'] = gpu_uncured_articles['Articles'].applymap(lemmatize_text)

    # uncured_articles = pd.read_table(path, header=None, names=['Articles'], skip_blank_lines=True)
    # uncured_articles['Tokenized Articles'] = uncured_articles['Articles'].apply(lemmatize_text)
    print(uncured_articles.head(20))
    uncured_articles['Tokenized Articles'].to_csv('feature_analysis_articles.csv', index=False)


if __name__ == '__main__':
    data_converter(MAIN_PATH + 'financial_articles.txt')