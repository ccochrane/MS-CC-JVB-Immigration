# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:15:23 2019

@author: chris
"""


# --------------------------------------------------------------------------
# Description
# --------------------------------------------------------------------------
'''
A script for training word2vec models on running 11-year segments of
words in Hansard by Conservative MPs.  

'''

#-----------------------------------------------------------------------------
# Initialization
#-----------------------------------------------------------------------------

import re

import pandas as pd
import nltk
import os
import numpy as np
import sys
from nltk.corpus import stopwords
import time

import gensim

from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import Phrases
import logging
import datetime

tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

stopwords = stopwords.words('english')


hansardSpeechesFull = pd.read_pickle('./lipad.pkl') 

def recodeParty(series):
    if series == 'Progressive Conservative':
        return 'Conservative'
    elif series == 'Conservative (1867-1942)':
        return 'Conservative'
    elif series == 'Laurier Liberal':
        return 'Liberal'
    elif series == 'Co-operative Commonwealth Federation (C.C.F.)':
        return 'NDP'
    elif series == 'Reform':
        return 'Conservative'
    elif series == 'Canadian Alliance':
        return 'Conservative'
    else:
        return series
      

hansardSpeechesFull['speakerparty'] = hansardSpeechesFull['speakerparty'].apply(recodeParty)

    



#Some function for converting Hansard to sentences, and sentences
#to wordlists.

def sentence_to_wordlist(sentence, remove_stopwords=True):
    sentence_text = re.sub(r'[^\w\s]','', sentence)
    words = sentence_text.lower().split()

    for word in words: #Remove Stopwords (Cochrane)
        if word in stopwords:
            words.remove(word)

    return words

def hansard_to_sentences(hansard, tokenizer, remove_stopwords=True ):
    #print("currently processing: word tokenizer")
    start_time = time.time()
    try:
        # 1. Use the NLTK tokenizer to split the text into sentences
        raw_sentences = tokenizer.tokenize(hansard.strip())
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call sentence_to_wordlist to get a list of words
                sentences.append(sentence_to_wordlist(raw_sentence))
        # 3. Return the list of sentences (each sentence is a list of words, so this returns a list of lists)
        len(sentences)
        return sentences
    except:
        print('nope')

    



for x in range(0,49):

    #What year is being processed
    startYear = 1956-x
    minYear = startYear-5
    maxYear = startYear+5
    #take 5 Hansard Speeches from 5 years before/after startYear
    hansardSpeechesDated = hansardSpeechesFull[(hansardSpeechesFull['speechdate'] > datetime.date(minYear,1,1)) & (hansardSpeechesFull['speechdate'] < datetime.date(maxYear,1,1)) & (hansardSpeechesFull['speakerparty']=='Conservative')]
    

    questions = hansardSpeechesDated['speechtext']
    
    questions = pd.Series.tolist(questions)
    sentences = []
    
    for i in range(0,len(questions)):
    
        start_time = time.time()
    
        try:
            # Need to first change "./." to "." so that sentences parse correctly
            hansard = questions[i].replace("/.", '')
            # Now apply functions
            sentences += hansard_to_sentences(hansard, tokenizer)
        except:
            print('no!')
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    
    num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count 
    num_workers = 4       # Number of threads to run in parallel
    context = 6           # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    
    from gensim.models import Phrases
    bigram_transformer = Phrases(sentences)
    
    
    model = word2vec.Word2Vec(bigram_transformer[sentences], workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    
    model.init_sims(replace=True)
    
    modelID = 'lipadConservative%d' % startYear
    
    model_name = modelID
    model.save(model_name)
