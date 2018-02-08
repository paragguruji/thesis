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

from time import time
from datetime import datetime
from random import sample

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


##############################################################################
import matplotlib
matplotlib.use('agg')

from pandas.plotting._misc import (scatter_matrix, radviz,
                                   andrews_curves, bootstrap_plot,
                                   parallel_coordinates, lag_plot,
                                   autocorrelation_plot)
from pandas.plotting._core import boxplot
from pandas.plotting._style import plot_params
from pandas.plotting._tools import table

from pandas.plotting._converter import \
    register as register_matplotlib_converters
from pandas.plotting._converter import \
    deregister as deregister_matplotlib_converters
##############################################################################

import sys
import os
import warnings
import logging
import json
import re

import multiprocessing as mp
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
logger = logging.getLogger(__name__)


def validate(d):
    if len(d['text']) < 280:
        return False
    return True


def preprocess(text):
    return ' '.join(map(lambda x: x.lower(),
                        re.split(r'[^a-zA-Z]+', text)))  # r'\W+'


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


def load_data(sources):
    t0 = time()
    _list = reduce(lambda x, y: x + y,
                   map(load_source_dicts, sources))
    logger.info("data loaded in %06.3fs\n" % (time() - t0))
    return pd.DataFrame.from_records(_list)


def wc(D, M, C):
    return sum([((D[M == i] - C[i])**2).sum(axis=1).sum(axis=0)
                for i in set(M)])


def _kmeans(D, K, max_iterations):
    C = np.array(sample(D, K))
    L_old = np.array([-1]*len(D))
    for _ in range(max_iterations):
        L = np.argmin(np.sqrt(((D - C[:, np.newaxis])**2).sum(axis=2)), axis=0)
        if all(L_old == L):
            break
        C = np.array(
                [D[L == l].mean(axis=0) if l in L else
                 np.array([np.inf]*D.shape[1])
                 for l in range(C.shape[0])])
        L_old = L.copy()
    return C, L


def experiment(X, K, max_iterations, run_count, run_timestamp='', _dir=None):
    if not run_timestamp:
        run_timestamp = \
            datetime.fromtimestamp(time()).strftime("%Y_%m_%d_%H_%M_%S")
    if not _dir:
        _dir = os.path.join('output', run_timestamp)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    km = MiniBatchKMeans(init='k-means++',
                         max_iter=max_iterations,
                         batch_size=100,
                         n_init=10,
                         max_no_improvement=10,
                         verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(len(K)):
            km.n_clusters = K[k]
            wcssds = []
            t00 = time()
            with open(os.path.join(_dir, 'K' + str(K[k]) + '.txt'), "w") as kf:
                kf.write(str(K[k]) + "\n")
                for run in range(run_count):
                    t0 = time()
                    km.fit(X)
                    wcssd = wc(X.toarray(),
                               km.labels_,
                               km.cluster_centers_)
                    wcssds.append(wcssd)
                    t1 = time()
                    logger.debug(("K: %03d  Run: %03d  WCSSD: %015.8f  " +
                                 "Time: %06.3fs") %
                                 (K[k], run, wcssd, (t1 - t0)))
                    kf.write("%s " % wcssd)
            t11 = time()
            logger.info(("K: %03d  Runs: %03d  Mean-WCSSD: %015.8f  " +
                         "Std. Dev.: %015.8f  Time: %06.3fs") %
                        (K[k],
                         run_count,
                         np.mean(np.array(wcssds)),
                         np.std(np.array(wcssds)),
                         (t11 - t00)))
    return K, run_timestamp, _dir


def analysis(K, run_timestamp='', _dir=None):
    result = np.array([0.0]*(len(K)*3)).reshape(len(K), 3)
    if not _dir:
        _dir = os.path.join('output', run_timestamp)
    wcssds = []
    for k in range(len(K)):
        try:
            with open(os.path.join(_dir, 'K' + str(K[k]) + '.txt'), "r") as kf:
                k_f = int(kf.next())
                if K[k] == k_f:
                    wcssds = map(float, kf.next().split())
                    result[k][0] = K[k]
                    result[k][1] = np.mean(wcssds)
                    result[k][2] = np.std(wcssds)
                else:
                    logger.fatal("Expected K: %s, found %s" % (K[k], k_f))
                    break
        except IOError:
            logger.error("IOError in results for K=%s" % K[k], exc_info=True)
    run_count = len(wcssds)
    df = pd.DataFrame(result, columns=['K', 'Mean-WC-SSD', 'STD-WC-SSD'])
    df.to_csv(os.path.join(_dir, 'results.csv'), index=False)
    return df, run_timestamp, run_count, _dir


def plot(df=None, run_timestamp='', run_count=0, _dir=None):
    if not _dir:
        _dir = os.path.join('output', run_timestamp)
    if df is None:
        df = pd.read_csv(os.path.join(_dir, 'results.csv'))
    p1 = df.plot(x=df.columns.values[0],
                 y=df.columns.values[1],
                 yerr=df.columns.values[2],
                 title="Mean WC-SSD for %s Runs Vs. K" % run_count)
    p1.set_xticks(df[df.columns.values[0]], minor=True)
    p1.grid(which='both', linestyle='dotted', alpha=0.5)
    p1.get_figure().savefig(os.path.join(_dir, 'Mean-WC-SSD.png'))


def build_setup(config=None):
    setup = {}
    if config is None:
        try:
            config = json.load(open('config.json'))
        except Exception:
            config = {}
            logger.error("config failed, using defaults", exc_info=True)

    df = load_data(config.get('sources',
                              ['cnn',
                               'foxnews',
                               'nytimes',
                               'nypost',
                               'bostonglobe',
                               'chicagotribune',
                               'latimes',
                               'wallstreetjournal',
                               'washingtonpost']))

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

    setup['X'] = vectorizer.fit_transform(df.text)
    setup['K'] = config.get('K', range(2, 201, 1))
    setup['K'] = range(*config.get('k_range', [2, 201, 1]))
    setup['max_iterations'] = config.get('max_iterations', 100)
    setup['run_count'] = config.get('run_count', 100)
    setup['timestamp'] = \
        datetime.fromtimestamp(time()).strftime("%Y_%m_%d_%H_%M_%S")

    formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logFilePath = os.path.join(os.getcwd(),
                               "log",
                               "thesis_kmeans_" + setup['timestamp'] + ".log")
    file_handler = logging.FileHandler(filename=logFilePath)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.ERROR)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Logging for experiment started at %s" % setup['timestamp'])

    return setup


# =============================================================================
# MAIN METHOD AND TESTING AREA
# =============================================================================


def main(config=None):
    """Description of main()"""
    setup = build_setup()
    t0 = time()
    experiment(setup['X'],
               setup['K'],
               max_iterations=setup['max_iterations'],
               run_count=setup['run_count'],
               run_timestamp=setup['timestamp'])

    plot(*analysis(K=setup['K'], run_timestamp=setup['timestamp']))
    logger.setLevel(logging.INFO)
    logger.info("Single process completed in %fs" % (time() - t0))


def multiprocess_main(config=None):
    setup = build_setup()
    pool = mp.Pool(mp.cpu_count())
    logger.setLevel(logging.ERROR)
    t0 = time()
    for k in setup['K']:
        pool.apply_async(experiment,
                         args=(setup['X'],
                               [k],
                               setup['max_iterations'],
                               setup['run_count'],
                               setup['timestamp']))

    pool.close()
    pool.join()
    logger.setLevel(logging.INFO)
    plot(*analysis(K=setup['K'], run_timestamp=setup['timestamp']))
    logger.info("Multiprocess completed in %fs" % (time() - t0))


if __name__ == '__main__':
    config = None
    if len(sys.argv) > 1:
        try:
            config = json.load(open(sys.argv[1]))
        except Exception:
            logger.error("Provided config file failed, trying default file")
#    main(config)
    multiprocess_main(config)
