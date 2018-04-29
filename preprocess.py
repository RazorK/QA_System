#!/sw/centos/anaconda3/5.0.1/bin/python3

from collections import defaultdict
import nltk
import os
import re
from nltk.stem.snowball import EnglishStemmer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from math import log
import pickle
from random import random
from time import time
from sys import exit
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer



tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
libstemmer = EnglishStemmer().stem

symbols = re.compile('''^[.,/?<>;':"[\]{}\\|!@#$%^&*()_+-=`~]*$''')
uni_stopwords = 'uni_stopwords.txt'

# def get_stopwords(filename):
#     stopwords = set()
#     with open(filename) as f:
#         for line in f:
#             line = libstemmer(line.strip())
#             stopwords.add(line)
#     # add more:
#     stopwords.add("'s")
#     stopwords.add('num')
#     stopwords.add('')
#     return stopwords

def stemmer(word):
    word = re.sub('''[.,/?<>;':"[\]{}\\|!@#$%^&*()_+-=`~]+$''', '', word)
    word = re.sub('''^[.,/?<>;':"[\]{}\\|!@#$%^&*()_+-=`~]+''', '', word)
    try:
        float(word)
        return 'num'
    except ValueError:
        if re.match(symbols, word):
            return ''
        else:
            return libstemmer(word)

def our_tokenizer(sentence):
    '''
    This is the tokenzier specifically designed for CountVectorizer.
    It will do stemming as well.
    '''
    return [stemmer(x) for x in tokenizer.tokenize(sentence) if stemmer(x)]


class Document():
    def __init__(self, idx, doc_type, content, peer_idx = None):
        self.idx = int(idx)
        self.content = content
        self.peer_idx = int(peer_idx) if peer_idx else None
    def __str__(self):
        return 'idx: %d; peer_idx: %s\n%s' % (self.idx, str(self.peer_idx), self.content)
    def get_content(self):
        return self.content


def build_document_list(files):
    docs = []
    for filename in files:
        with open(filename, encoding='utf-8') as f:
            f.readline()
            for line in f:
                docs.append(Document(*line.split(',')))
    return docs

def cosine_similarity(v1, v2):
    assert len(v1) == len(v2)
    dot_product = sum([x * y for x, y in zip(v1, v2)])
    norm_v1 = sum(x**2 for x in v1)**0.5
    norm_v2 = sum(x**2 for x in v2)**0.5
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product/norm_v1/norm_v2

def generate_count_vectorizer():
    stopwords =  pickle.load(open('stopwords.p', 'rb'))
    cv = CountVectorizer(tokenizer = our_tokenizer, stop_words = stopwords)
    
    # anwwer files:
    files = ['./stack_exchange/' + x for x in os.listdir('./stack_exchange/') if 'answers' in x]
    answers = build_document_list(files)
    X = cv.fit_transform([d.get_content() for d in answers]).toarray()
    word_ratio = [sum(x)/len(X) for x in zip(*X)]
    return X, cv, answers, word_ratio

# cv, X, answers = generate_count_vectorizer()
# pickle.dump(cv, open('cv.p', 'wb'))
# pickle.dump(X, open('X.p', 'wb'))
# pickle.dump(answers, open('answers.p', 'wb'))

# to load them, run pickle.load(). E.g., cv = pickle.load(open('cv.p', 'rb'))
# generate_count_vectorizer()

