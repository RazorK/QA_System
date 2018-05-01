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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups


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
    def __init__(self, line):
        components = line.split(',')
        if len(components) == 3:
            idx, _, content = components
            self.peer_idx = None
        elif len(components) == 4:
            idx, content, _, peer_idx = components
            self.peer_idx = int(peer_idx)
        self.idx = int(idx)
        self.content = content
        
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
                docs.append(Document(line))
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
    cv = CountVectorizer(tokenizer = our_tokenizer, stop_words = stopwords, ngram_range=(1, 2))
    
    # answer files:
    answer_files = ['./stack_exchange/' + x for x in os.listdir('./stack_exchange/') if 'answers' in x]
    # question files
    question_files = ['./stack_exchange/' + x for x in os.listdir('./stack_exchange/') if 'questions' in x]
    answers = build_document_list(answer_files)
    questions = build_document_list(question_files)
    # count vector for answers
    answer_matrix = cv.fit_transform([d.get_content() for d in answers])
    num_answers, num_features = answer_matrix.shape
    answer_mapping = {answer.idx : i for i, answer in enumerate(answers)}

    word_ratio = [feature.sum()/num_answers for feature in answer_matrix.transpose()]

    return answer_matrix, cv, answers, questions, word_ratio, answer_mapping


def count_nonzero_features(sparse_matrix):
    lengths = [x.count_nonzero() for x in sparse_matrix]
    def length_of(n):
        return len([1 for x in lengths if x > n])
    for i in range(10, 100, 10):
        print('length > %d: %d' % (i, length_of(i)))


def generate_cos_sim_rank_list(questions, answers, answer_mapping, converter):
    '''
    parameters:
        questions       -   list of list
        answers         -   list of list
        answer_mapping  -   dict of int : int
        converter       -   a function to convert question string into a vector
    return:
        list of ranks
    '''
    results = []
    for q in questions:
        question = converter(q)
        l = [cosine_similarity(question, a) for a in answers]
        ranks = sorted(list(range(len(l))), key = lambda i : l[i], reverse = True)
        target = answer_mapping[q.peer_idx]
        results.append(ranks.index(target))
    return results




def generate_rank_range(ranks, length, scale = 5):
    ranks = sorted(ranks)
    c = 1
    total = 0
    result = [[i, 1] for i in range(0, length + 1, scale)]
    result[0] = [0, 0]
    for x in ranks:
        if x < c * scale:
            total += 1
        else:
#             print('prepare', (c*scale, total/length), ' after hitting', x)
            result[c] = [c*scale, total/length]
            c += 1
    return result
    

        


def init_lda(num_topics, max_iterations):
    answer_matrix, cv, answers, questions, word_ratio, answer_mapping = generate_count_vectorizer()

#     lda = LatentDirichletAllocation(n_components=num_topics, max_iter=max_iterations)
#     lda.fit(answer_matrix)
# 
#     pickle.dump(lda, open('simple_lda.p','wb'))
# 
#     count_nonzero_features(answer_matrix)
# 

    lda = pickle.load(open('simple_lda.p', 'rb'))
    print('loading done')

    
    
#     def get_best_similarity(question, answer_matrix, lda, word_ratio, lam, miu):
#         '''
#         question    -   sparse matrix representing the question
#         '''
#         best = (0, 0)
#         simiList = []
#         for answer in answer_matrix:
#             prob = 1
#             answerDis = lda.transform(answer)
#             for i in np.nditer(question.indices):
#                 pseudo = (answer[0, i] + miu * word_ratio[i]) / (miu + answer.sum())
#                 plda = 0
#                 for topic_idx, topic in enumerate(lda.components_):
#                     plda += answerDis[0, topic_idx] * topic.item(i) * question[0, i]
#                 prob *= lam * pseudo + (1-lam) * plda
#             simiList.append(prob)
#         print('after ', time() - start, ' seconds, start to sort')
#         res = list(range(len(simiList)))
#         return sorted(res, key = lambda i : simiList[i], reverse= True)
    

    def getQLPSortedIndexList(qes, tf, answerDis, lda, cwlist, lam, u):
    # should return list of answer index
        simiList = []
        for j, row in enumerate(tf):
            prob = 1

            for i in range(len(qes)):
                if(qes[i] == 0): continue
                pseudo = (row[i] + u * cwlist[i]) / (u + sum(row))
            
                # calculate plda
                plda = 0
            
                # traverse topic
                for topic_idx, topic in enumerate(lda.components_):
                    plda += answerDistributions[j][topic_idx] * topic.item(i)*qes[i]
                
                prob *= lam*pseudo + (1-lam) * plda
        
            simiList.append(prob)
        # sort the similarity list, and return the index list.
        res = list(range(len(simiList)))
        return sorted(res, key = lambda i : simiList[i], reverse= True)


    print('start!!')


    answerDistributions = lda.transform(answer_matrix).tolist()
    converter = lambda q : lda.transform(cv.transform([q.content])).tolist()[0]
    return generate_cos_sim_rank_list(questions, answerDistributions, answer_mapping, converter)




#     X = answer_matrix.toarray()
#     converter = lambda q: cv.transform([q.content]).toarray()[0]
#     return generate_cos_sim_rank_list(questions, X, answer_mapping, converter)

#     for q in questions:
#         start = time()
#         question = cv.transform([q.content])
#         # print('question shape: ', question.shape, question.count_nonzero())
#         print(question[0,:].count_nonzero(), ' tokens;', end=' ')
#         question = question[0].toarray()[0]
#         l = getQLPSortedIndexList(question, X, answerDistributions, lda, word_ratio, 0.5, 0.5)
#         target = answer_mapping[q.peer_idx]
#         print('rank: ', l.index(target), '; cost', time() - start, 'seconds;', )
# 


# 

    


#     start = time()
#     
# 
#     print(get_best_similarity(question, answer_matrix, lda, word_ratio, 0, 1))
#     print('cost ', time() - start, ' seconds')
# 



results = pickle.load(open('tfidf_res.p', 'rb'))
# results = init_lda(51, 300)
print(results)


def plot_range(results):
    '''
    results -   list
    you may want to change the label to something you want
    '''
    rank_range = generate_rank_range(results, len(results), 10)
    rank_range_X, rank_range_Y = list(zip(*rank_range))
    plt.plot(rank_range_X, rank_range_Y, label=('curve'))
    plt.legend()
    plt.title("cumulative range")
    plt.xlabel('ranges')
    plt.ylabel('predictions')

    plt.show()
        
plot_range(results)

# pickle.dump(cv, open('cv.p', 'wb'))
# pickle.dump(X, open('X.p', 'wb'))
# pickle.dump(answers, open('answers.p', 'wb'))

# to load them, run pickle.load(). E.g., cv = pickle.load(open('cv.p', 'rb'))
# generate_count_vectorizer()



