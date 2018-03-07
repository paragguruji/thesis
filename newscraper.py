# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:40:40 2018

@author: Parag
"""

import csv
import json
import traceback
from newspaper import Article

sources = json.load(open('config.json'))['sources']

for source in sources:
    article_index = []
    with open('data/' + source + '/index.csv', 'r') as fp:
        reader = csv.reader(fp)
        article_index = [r for r in reader]

    for a in article_index:
        try:
            article = Article(a[1], language='en', fetch_images=False)
            article.download()
            article.parse()
            article.nlp()
            d = {'url': a[1],
                 'id': a[2],
                 'title': article.title,
                 'text': article.text,
                 'authors': article.authors,
                 'publish_date': article.publish_date.date().isoformat(),
                 'source': source,
                 'top_words': article.keywords,
                 'summary': article.summary}
            with open('data/' + source + '/new' + a[2] + '.json', 'w') as f:
                json.dump(d, f)
        except Exception:
            print("Exception at " + a[2])
            traceback.print_exc()


def flatten_auths(auth):
    if ' and ' in auth:
        return map(lambda x: x.strip(), auth.split(' and '))
    if ',' in auth:
        return map(lambda x: x.strip(), auth.split(','))
    return [auth]


def sep_auth(auth):
    auth = flatten_auths(auth)
    while True:
        done = True
        for a in auth:
            if ' and ' in a or ',' in a:
                done = False
                _list = flatten_auths(a)
                auth.remove(a)
                auth.extend(_list)
        if done:
            break
    return auth

tot = reduce(lambda x, y: x + y, total_authors)
