#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summary

description

:REQUIRES:

:TODO:

:AUTHOR: Parag Guruji
:ORGANIZATION: Purdue University
:CONTACT: pguruji@purdue.edu
:SINCE: Thu Dec 14 17:47:00 2017
:VERSION: 0.1
"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

from time import time
from datetime import datetime
from random import sample

import os
import json
import re
import logging
import numpy as np
import pandas as pd




# =============================================================================
# PROGRAM METADATA
# =============================================================================
__author__ = 'Parag Guruji'
__contact__ = 'pguruji@purdue.edu'
__copyright__ = 'Copyright (C) 2017 Parag Guruji, Purdue University, USA'
__date__ = 'Thu Dec 14 17:47:00 2017'
__version__ = '0.1'


# =============================================================================
# CLASSES / METHODS
# =============================================================================
DATA_DIR = os.path.join(os.getcwd(), "data")
MAX_ITERATIONS = 50;

def validate(d):
    if len(d['text']) < 280:
        return False
    return True


def preprocess(text):
    return ' '.join(map(lambda x: x.lower(), re.split(r'[^a-zA-Z]+', text))) # r'\W+'


def load_source_dicts(source):
    _dicts = [json.load(open(os.path.join(DATA_DIR, source, f)))
              for f in os.listdir(os.path.join(DATA_DIR, source))
              if f.endswith('.dict')]
    valid_dicts = []
    for d in _dicts:
        if validate(d):
            d['text'] = preprocess(d['text'])
            d['source'] = source
            valid_dicts.append(d)
    return valid_dicts


def load_data():
    _list = reduce(lambda x, y: x + y,
                   map(load_source_dicts, ['cnn',
                                           'foxnews',
                                           'nytimes',
                                           'nypost',
                                           'bostonglobe',
                                           'chicagotribune',
                                           'latimes',
                                           'wallstreetjournal',
                                           'washingtonpost']))
    return pd.DataFrame.from_records(_list)
    # load_source('cnn') + load_source('foxnews')
    # return pd.DataFrame({'text':[preprocess_article(d['text']) for d in _list]})

#lambda x: " ".join(re.findall(r'\w+', x))

t0 = time()
vectorizer = TfidfVectorizer(
                max_df=0.1,
                max_features=1200,
                min_df=1,
                stop_words='english',
                preprocessor=preprocess,
                use_idf=True,
                analyzer='word',
                ngram_range=(2, 3))

km = MiniBatchKMeans(init='k-means++',
                     n_clusters=4,
                     batch_size=100,
                     n_init=10,
                     max_no_improvement=10,
                     verbose=0)


D = vectorizer.fit_transform(load_data().text)

#X = vectorizer.fit_transform(load_data().text)


K_CONST = range(2,151,1)

print("done in %fs" % (time() - t0))
print()




def wc(D, M, C):
    return sum([((D[M == i] - C[i])**2).sum(axis=1).sum(axis=0)
                for i in set(M)])

def _kmeans(D, K):
    C = np.array(sample(D, K))
    L_old = np.array([-1]*len(D))
    for _ in range(MAX_ITERATIONS):
        L = np.argmin(np.sqrt(((D - C[:, np.newaxis])**2).sum(axis=2)), axis=0)
        if all(L_old == L):
            break
        C = np.array(
                [D[L == l].mean(axis=0) if l in L else
                 np.array([np.inf]*D.shape[1])
                 for l in range(C.shape[0])])
        L_old = L.copy()
    return C, L


def analysis(_dir=None, algo='sklearn', K=K_CONST):
    if not _dir:
        _dir = os.path.join('output', 'analysis')
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    result = np.array([0.0]*(len(K)*3)).reshape(len(K), 3)
    for run in range(100):
        for k in range(len(K)):
            result[k][0] = K[k]
#            1. sklearn k-means:
            if algo == 'sklearn':
                km.n_clusters = K[k]
                print("Clustering sparse data with %s" % km)
                t0 = time()
                km.fit(D)
#                print("done in %0.3fs" % (time() - t0))
#                print()
                result[k][1] += wc(D.toarray(),
                                   km.labels_,
                                   km.cluster_centers_)
                result[k][2] += metrics.silhouette_score(D,
                                                         km.labels_,
                                                         sample_size=1000)
#            2. My k-means:
            else:
                print("Clustering data with Parag's K Means: K=%s" % K[k])
                t0 = time()
                C, M = _kmeans(D.toarray(), K[k])
                print("done in %0.3fs" % (time() - t0))
                print()
                result[k][1] = wc(D.toarray(), M, C)
                result[k][2] = metrics.silhouette_score(D, M, sample_size=100)

    result[:, [1, 2]] /= 100.0

    df = pd.DataFrame(result, columns=['K', 'WC-SSD', 'SC'])
    run_timestamp = datetime.fromtimestamp(time()).strftime("%Y%m%d%H%M%S")
    df.to_csv(os.path.join(_dir, 'Run' + run_timestamp + '.csv'), index=False)
    plot_b1(_dir, df, run_timestamp)
    return df


def plot_b1(_dir=None, df=None, timestamp=''):
    if not _dir:
        _dir = os.path.join('output', 'analysis')
    if df is None:
        df = pd.read_csv(os.path.join(_dir, 'Run' + timestamp + '.csv'))
    p1 = df.plot(x=df.columns.values[0],
                 y=df.columns.values[1],
                 title="Analysis : Variation in WC-SSD with K")
    p1.set_xticks(df[df.columns.values[0]], minor=True)
    p1.grid(which='both', linestyle='dotted', alpha=0.5)
    p1.get_figure().savefig(os.path.join(_dir, 'WC-SSD' + timestamp + '.png'))

    p2 = df.plot(x=df.columns.values[0],
                 y=df.columns.values[2],
                 title="Analysis B.1: Variation in SC with K")
    p2.set_xticks(df[df.columns.values[0]], minor=True)
    p2.grid(which='both', linestyle='dotted', alpha=0.5)
    p2.get_figure().savefig(os.path.join(_dir, 'SC' + timestamp + '.png'))

#
#order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#
#terms = vectorizer.get_feature_names()
#for i in range(3):
#    print("Cluster %d:" % i, end='')
#    for ind in order_centroids[i, :10]:
#        print(' %s' % terms[ind], end='')
#    print()
#



# =============================================================================
# MAIN METHOD AND TESTING AREA
# =============================================================================


def main():
    """Description of main()"""
    #analysis()
    plot_b1()

if __name__ == '__main__':
    main()

