# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:25:24 2018

@author: Parag
"""
import json
import pandas as pd
import re
import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


def preprocess(text):
    """Preprocesses given text and returns preprocessed text
    Strips everything but alphabets and converts to lowercase



    :param text: (*string*) text to preprocess
    :paaram returns: (*string*) preprocessed text


    """
    return ' '.join(map(lambda x: x.lower(),
                        re.split(r'[^a-zA-Z]+', text)))  # r'\W+'


config = json.load(open('config.json'))
_list = json.load(open('data/lowercased_author_present_dicts.json', 'r'))
for i in range(len(_list)):
    _list[i]['uid'] = i+1
df = pd.DataFrame.from_records(_list)

vectorizer = TfidfVectorizer(max_df=config.get('max_df', 0.1),
                             max_features=config.get('max_features', 1200),
                             min_df=config.get('min_df', 1),
                             stop_words=config.get('stop_words',
                                                   'english'),
                             preprocessor=preprocess,
                             use_idf=config.get('use_idf', True),
                             analyzer=config.get('analyzer', 'word'),
                             ngram_range=tuple(config.get('ngram_range',
                                                          (2, 3))))

X = vectorizer.fit_transform(df.text)

scores = zip(vectorizer.get_feature_names(), np.asarray(X.sum(axis=0)).ravel())
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
# for item in sorted_scores:
#     print "{0:50} Score: {1}".format(item[0], item[1])

K = [700]

km = MiniBatchKMeans(init='k-means++',
                     n_clusters=700,
                     max_iter=30,
                     reassignment_ratio=0.0,
                     batch_size=2000,
                     init_size=1000,
                     n_init=1,
                     max_no_improvement=30,
                     verbose=True)
km.fit(X)
actual_clusters = list(set(list(km.labels_)))

cluster_size_distribution = Counter(km.labels_)
dfClusterSizes = pd.DataFrame(cluster_size_distribution.items(),
                              columns=['cluster', 'article_count'])
dfClusterSizes_sorted = dfClusterSizes.sort_values(by='article_count',
                                                   ascending=False)
dfClusterSizes_sorted.reset_index(drop=True)
dfClusterSizes_sorted.plot(kind='bar')


author_distribution = json.load(open('data/author_distribution.json'))
dfAuthors = pd.DataFrame(author_distribution.items(),
                         columns=['author', 'article_count'])
dfAuthors_sorted = dfAuthors.sort_values(by='article_count', ascending=False)
dfAuthors_sorted.reset_index(drop=True)
dfAuthors_sorted.plot(kind='bar')
