#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:54:42 2019

@author: chris cochrane
"""

# --------------------------------------------------------------------------
# Description
# --------------------------------------------------------------------------
'''
Script for storing Liberal Party results in data.frame format.

'''


#-----------------------------------------------------------------------------
# Initialization
#-----------------------------------------------------------------------------


import gensim
import numpy as np
from operator import itemgetter
import pylab as pl
import scipy.stats as stats
import nltk
import time
import pandas as pd
import re
from nltk.corpus import stopwords


#-----------------------------------------------------------------------------
# Creating Empty Pandas DataFrame
#-----------------------------------------------------------------------------

df = pd.DataFrame(columns=['word',*[x for x in range(1908,2013)]])

#-----------------------------------------------------------------------------
# Loading stored w2v Models for each year by looping over available years
#-----------------------------------------------------------------------------

for year in range(1908,2013):
    modelName = 'lipadLiberal'+str(year)
    model = gensim.models.Word2Vec.load(modelName)


#-----------------------------------------------------------------------------
# Seed Words
#-----------------------------------------------------------------------------

    immigrants = model.wv['immigrants']
    immigration = model.wv['immigration']
    
    
    
    vocab = list(model.wv.vocab.keys()) #the full vocabulary of Google News
    
    
    # weights and words
    
    runningTally=[]
    dictOfWeights = {}
    
    #-----------------------------------------------------------------------------
    # Model
    #-----------------------------------------------------------------------------
    
    '''for every word in the hansard, calculate its cosine similarity to the 
    lists of positive words and negative words, then substract the sum of that
    word's cosine simlarity to negative seed words from its cosine similarity to the
    postive seed words.''' 
    
    for word in vocab:
    
        word_model = model.wv[word]
    
        immig1 = np.dot(word_model, immigrants) / (np.linalg.norm(word_model) * np.linalg.norm(immigrants))
        immig2 = np.dot(word_model, immigration) / (np.linalg.norm(word_model) * np.linalg.norm(immigration))
    
        immig = sum([immig1, immig2])
    
        result = (word, immig)
        runningTally.append(result)
        dictOfWeights[word] = result
    
    
    
    #-----------------------------------------------------------------------------
    # Results
    #-----------------------------------------------------------------------------
    
    '''The 100 most positive signed and most negative signed words'''
    
    runningTally = sorted(runningTally, key=itemgetter(1), reverse=True)
    print("Top Relevant:", runningTally[:25])
    print("Total Vocabulary Size in ", year,": ", len(vocab))
    for item in runningTally[:2500]:
        df.loc[item[0], year] = item[1]


#-----------------------------------------------------------------------------
# Results
#-----------------------------------------------------------------------------
    

df.to_csv('wordCosinesliberalsfinal.csv', sep=',')
        