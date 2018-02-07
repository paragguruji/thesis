#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scrape the news article in bulk

Scrape news articles using python package newspaper.
Filter the articles relevant to travel ban based on URL and keywords.
Define a data structure to save text and other components (title, date, etc.).
Store articles in folders per news agency and file name as date + title


:REQUIRES:
    newspaper, Index.csv file containing Tilte, URL, Idx columns

:TODO:
    Done


:AUTHOR: Parag Guruji
:ORGANIZATION: Purdue University
:CONTACT: pguruji@purdue.edu
:SINCE: Tue Aug 22 20:01:00 2017
:VERSION: 0.1
"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================
# from xgoogle.search import GoogleSearch, SearchError
from newspaper import Article
import sys
import os
import csv
import json
import datetime

# =============================================================================
# PROGRAM METADATA
# =============================================================================
__author__ = 'Parag Guruji'
__contact__ = 'pguruji@purdue.edu'
__copyright__ = 'Copyright (C) 2017 Parag Guruji, Purdue University, USA'
__date__ = 'Tue Aug 22 20:01:00 2017'
__version__ = '0.1'


# =============================================================================
# CLASSES / METHODS
# =============================================================================

# Failed code for google search - failed due to Google API rules
#    try:
#        gs = GoogleSearch("trump travel ban site:cnn.com")
#        gs.results_per_page = 100
#        results = []
#        while True:
#            print "going in"
#            page = gs.get_results()
#            if not page:
#                break
#            results.extend(page)
#        print "No. of results: ", len(results)
#        print results[:10]
#    except SearchError, e:
#        print "Search Failed: %s" % e
#

DATA_DIR = os.path.join(os.getcwd(), "data")


def scrapeArticle(url):
    """Scrapes article from specified url and translates into standard dict
    """
    d = {}
    article = Article(url, language='en', fetch_images=False)
    article.download()
    if article.is_downloaded:
        article.parse()
        if article.is_parsed:
            d['url'] = url
            if 'meta_data' in article.__dict__:
                if 'og' in article.__dict__['meta_data']:
                    if 'title' in article.__dict__['meta_data']['og']:
                        d['title'] = \
                            article.__dict__['meta_data']['og']['title']
                    if 'type' in article.__dict__['meta_data']['og']:
                        d['type'] = \
                            article.__dict__['meta_data']['og']['type']
                    if 'site_name' in article.__dict__['meta_data']['og']:
                        d['source'] = \
                            article.__dict__['meta_data']['og']['site_name']
                if 'author' in article.__dict__['meta_data']:
                    d['authors'] = article.__dict__['meta_data']['author']
                if 'section' in article.__dict__['meta_data']:
                    d['section'] = article.__dict__['meta_data']['section']
            if 'publish_date' in article.__dict__:
                if isinstance(article.__dict__['publish_date'],
                              datetime.datetime):
                    d['publish_date'] = \
                        article.__dict__['publish_date'].date().isoformat()
            if 'text' in article.__dict__:
                d['text'] = article.text
    return d


def saveArticle(articleDict, path):
    """Saves dict representing an article as a json file at the given path
    """
    with open(path, 'w') as fp:
        json.dump(articleDict, fp)


def retryFailed(source_name):
    """Retry scraping articles from given source which have id -1
    """
    article_index = []
    idx = 0
    with open(os.path.join(DATA_DIR, source_name, "index.csv"),
              "r") as fp:
        reader = csv.reader(fp)
        header = reader.next()
        article_index = [r for r in reader]
        idx = max(map(lambda x: int(x) if x else 0,
                      [r[2] for r in article_index]))
    for record in article_index:
        if record[2] in [-1, '-1']:
            print "Retrying article: ", record[0], " from: ", record[1]
            d = scrapeArticle(record[1].strip())
            if d and \
               'publish_date' in d and \
               d['publish_date'].startswith('2017'):
                idx += 1
                saveArticle(d, os.path.join(DATA_DIR,
                                            source_name,
                                            str(idx) + ".dict"))
                record[2] = idx
            else:
                record[2] = -1
    print "Saving new index"
    with open(os.path.join(DATA_DIR, source_name, "index.csv"),
              "wb") as fp:
        writer = csv.writer(fp)
        writer.writerows([header] + article_index)


def getArticles(source_name):
    """Reads index.csv to scrape and save articles saves them with a serial\
    no.. Then modifies index to show the serial no."""
    article_index = []
    idx = 0
    with open(os.path.join(DATA_DIR, source_name, "Manual list.csv"),
              "r") as fp:
        reader = csv.reader(fp)
        header = reader.next()
        article_index = [r for r in reader]
    for record in article_index:
        if record[2].strip() not in ['0', 0, '']:
            if record[2] != -1:
                idx = record[2]
            continue
        else:
            d = scrapeArticle(record[1].strip())
            if d and \
               'publish_date' in d and \
               d['publish_date'].startswith('2017'):
                idx += 1
                saveArticle(d, os.path.join(DATA_DIR,
                                            source_name,
                                            str(idx) + ".dict"))
                record[2] = idx
            else:
                record[2] = -1
    print "Saving new index"
    with open(os.path.join(DATA_DIR, source_name, "index.csv"),
              "wb") as fp:
        writer = csv.writer(fp)
        writer.writerows([header] + article_index)


# =============================================================================
# MAIN METHOD AND TESTING AREA
# =============================================================================
def main():
    """Controller of flow"""
    print "argv: ", sys.argv
    sys.stdout.flush()
    if len(sys.argv) > 2 and sys.argv[2] == '-r':
        print "Retrying to scrape failed articles from", sys.argv[1]
        sys.stdout.flush()
        retryFailed(sys.argv[1])
    else:
        print "Scraping articles from", sys.argv[1]
        sys.stdout.flush()
        getArticles(sys.argv[1])


if __name__ == '__main__':
    main()
