from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

MAIN_PATH = 'C:\\\\Users\\cmazz\\PycharmProjects\\TextClassification\\'

class KmeansCluster:
    def __init__(self, document):
        self.document = document.to_list()

    def vectorize(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(self.document)

    def create_fit_model(self, true_k=3, init='k-means++', max_iter=100, n_init=1):
        model = KMeans(n_clusters=true_k, init=init, max_iter=max_iter, n_init=n_init)
        self.model = model.fit(self.X)

    def cluster_form(self):
        self.order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        self.terms = self.vectorizer.get_feature_names()

    def predict(self, list_article):
        self.predict = self.vectorizer.transform(list_article)
        return self.model.predict(self.predict)


if __name__=='__main__':
    path = MAIN_PATH + 'financial_articles_special_char.txt'
    uncured_articles = pd.read_table(path, header=None, names=['Articles'], skip_blank_lines=True)
    kmeans = KmeansCluster(uncured_articles['Articles'].tail(30))
    kmeans.vectorize()
    true_k = 10
    kmeans.create_fit_model(true_k=true_k)
    kmeans.cluster_form()
    for i in range(true_k):
        print("Cluster {}:".format(i)),
        for ind in kmeans.order_centroids[i, :10]:
            print(kmeans.terms[ind])
    prediction = kmeans.predict()
