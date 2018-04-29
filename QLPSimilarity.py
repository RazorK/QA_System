from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import pickle
from preprocess import *

def getQLPSortedIndexList(question, cv, tf, lda,cwlist, lam, u):
    # should return list of answer index
    simiList = []
    qes = cv.transform([question.content])[0].toarray()[0]
    for row in tf:
        prob = 1
        answerDis = lda.transform([row]).tolist()[0]
        for i in range(len(qes)):
            if(qes[i] == 0): continue
            pseudo = (row[i] + u * cwlist[i]) / (u + sum(row))
            
            # calculate plda
            plda = 0
            
            # traverse topic
            for topic_idx, topic in enumerate(lda.components_):
                plda += answerDis[topic_idx] * topic.item(i)*qes[i]
                
            prob *= lam*pseudo + (1-lam) * plda
        
        simiList.append(prob)
    # sort the similarity list, and return the index list.
    res = list(range(len(simiList)))
    return sorted(res, key = lambda i : simiList[i], reverse= True)
    

n_components = 51
max_it = 300


X, cv, answers, word_ratio = generate_count_vectorizer()

lda = LatentDirichletAllocation(n_components=n_components, max_iter=max_it,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(X)

ques = answers[17]

l = getQLPSortedIndexList(ques, cv, X, lda, word_ratio, 0, 1)
print(l)