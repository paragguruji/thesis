# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:16:53 2018
@author: lenovo
"""
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import pandas as pd
import json
from random import shuffle
import logging
import os.path
import sys
import multiprocessing

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, _list):
        # self.sources = sources
        # flipped = {}
        # make sure that keys are unique
        self.sentences = []
        for i in range(len(_list)):
            self.sentences.append(LabeledSentence(
                        utils.to_unicode(_list[i]).split(),
                        ['news_article' + '_%s' % i]))
#
#        for key, value in sources.items():
#            if value not in flipped:
#                flipped[value] = [key]
#            else:
#                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(),
                                          [prefix + '_%s' % item_no])

    def to_array(self):
#        for source, prefix in self.sources.items():
#            with utils.smart_open(source) as fin:
#                for item_no, line in enumerate(fin):
#                    self.sentences.append(LabeledSentence(
#                        utils.to_unicode(line).split(),
#                        [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


_list = json.load(open('data/lowercased_author_present_dicts.json', 'r'))
for i in range(len(_list)):
    _list[i]['uid'] = i+1

df = pd.DataFrame.from_records(_list)
doc2vecData = [d.replace('\n', ' ') for d in df.text.tolist()]
sentences = LabeledLineSentence(doc2vecData)

model = Doc2Vec(min_count=1,
                window=5,
                size=100,
                sample=1e-4,
                negative=5,
                workers=multiprocessing.cpu_count)
'''
min_count: ignore all words with total frequency lower than this.
You have to set this to 1, since the sentence labels only appear once.
Setting it any higher than 1 will miss out on the sentences.

window: the maximum distance between the current and predicted word within
a sentence. Word2Vec uses a skip-gram model, and this is simply the window
size of the skip-gram model.

size: dimensionality of the feature vectors in output. 100 is a good number.
If you're extreme, you can go up to around 400.

sample: threshold for configuring which higher-frequency words are randomly
downsampled

workers: use this many worker threads to train the model
'''
model.build_vocab(sentences.to_array())

for epoch in range(50):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm(),
                total_examples=model.corpus_count,
                epochs=model.iter,)

model.save('./NewsArticles.d2v')