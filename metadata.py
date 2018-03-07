# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:43:17 2018

@author: Parag
"""

from experiments import load_data
import json

config = json.load(open('config.json'))

base_df = load_data(config['sources'])

auth = base_df['authors'].to_dict().values()


def flatten_auths(auth):
    if ' and ' in auth:
        return map(lambda x: x.strip(), auth.split(' and '))
    if ',' in auth:
        return map(lambda x: x.strip(), auth.split(','))
    return [auth]


def sep_auth(auth):
    auth = flatten_auths(auth)
    _list = []
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


unique_authors = list(set(reduce(lambda x, y: x + y, sep_auth(auth))))

clean_authors = [t for t in unique_authors if t and
                 'CNN' not in t and
                 '2017' not in t and
                 'The ' not in t and
                 'facebook.com' not in t]

for a in ['Wire',
          'TIME',
          'Anchor',
          'Associated Press',
          'Baltimore Sun',
          'Chicago Tribune',
          'Chief National Security Correspondent',
          'Foreign Policy',
          'Fortune Editors',
          'Fortune Video',
          'Los Angeles Times',
          'National Security Producer',
          'Staff Report',
          'Supreme Court Biographer',
          'Times Staff',
          'Tribune news services',
          'U-T Letters',
          'tronc video',
          'A Statement Oxfam',
          'About New York',
          'Acapulco Travel Office In Lake Forest',
          'April',
          'Anchor',
          'Associated Press',
          'August',
          'California Today',
          'A Statement Oxfam',
          'About New York',
          'Acapulco Travel Office In Lake Forest',
          'Anchor',
          'April',
          'Associated Press',
          'August',
          'California Today',
          'Chief Executive Of Discover Los Angele',
          'Chief National Security Correspondent',
          'Circuit Court Of Appeal',
          'Cnn',
          'Cnn Editor-At-Large',
          'Cnn Legal Analyst',
          'Cnn National Security Analyst',
          'Cnn Photographs Melissa Golden',
          'Cnn Polling Director',
          'Cnn Religion Editor',
          'Cnn Senior Un Correspondent',
          'Cnn Supreme Court Reporter',
          'Cnn White House Producer',
          'Director Of The International Refugee Assistance Project',
          'February',
          'Fortune Editors',
          'Fortune Video',
          'Fox News',
          'General Manager',
          'I Was Misinformed',
          'Insider Podcasts',
          'J. Paul Getty Trust President',
          'January',
          'July',
          'June',
          'Los Angeles City Controller',
          'Los Angeles Producer Rossi Canno',
          'March',
          'May',
          'New America Foundation',
          'New England Journal Of Medicine',
          'New York Today',
          'On Washington',
          'Post Editorial Board',
          'Post Wire Report',
          'President Trum',
          'Retro Report',
          'September',
          'Staff Report',
          'State Of The Art',
          'Supreme Court Biographer',
          'The Associated Press',
          'The Carpetbagger',
          'The Daily',
          'The Editorial Board',
          'The Ethicist',
          'The Getaway',
          'The Interpreter',
          'The New York Times',
          'The San Diego Union-Tribune Editorial Board',
          'The Times Editorial Board',
          'This Land',
          'Tribune News Services',
          'Video Claudia Morales',
          'Washington State Attorney General',
          'White House Correspondent',
          'Wire',
          'Witw Staff']:
    if a in clean_authors:
        clean_authors.remove(a)

targets = [t for t in clean_authors if ' by ' in t]
for t in targets:
    if any(lambda x: x in t.lower(), ['video by',
                                      'photos by',
                                      'photographs by',
                                      'photography by']):
        pass
    else:
        author = t.split(' by ')[-1]
        clean_authors.append(author)
    clean_authors.remove(t)

clean_authors.sort()
