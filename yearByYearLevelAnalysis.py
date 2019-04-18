#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 07:31:00 2019

@author: chriscochrane
"""
# --------------------------------------------------------------------------
# Description
# --------------------------------------------------------------------------
'''
This script loads word2vec models which have been trained on 11-year moving
segments of Hansard from 1908-2013, seperately for the Liberals and the 
Conservatives. The analyses examines word usage around the two words
"immigration" and "immigrants."  In particular:
    
    1) the frequency of mentions of immigration/immigrants by party in each
    11-year window.
    
    2) The words with cosine similarities closest to immigrants/immigration,
    for both Liberals and Conservatives.
    
    3) The valence of immigrants/immigration, measured both in terms of the 
    cosine similarities of these words to sets of postive and negative seed
    words, and also in terms of the valence of all 2500 words most closely
    linked to immigrants/immigration for each party in each window.
    
 The basic questions:
        1) How has the level of discussion relating to immigration changed 
    over time?
        2) Do Conservatives and Liberals discuss immigration using the same
    kinds of concepts, and has this changed over time?
        3) Is immigration valenced the same way by Conservatives and Liberals,
    and has this changed over time?
        
That data are organized by party/year.  

'''


# --------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------

import pandas as pd
import numpy as np
from operator import itemgetter
import gensim

# --------------------------------------------------------------------------
# loading Data
# --------------------------------------------------------------------------

#The Top 2500 words for each party/year associated with immigrants/immigration
conservativeCosines = pd.read_csv("wordCosinesConservatives.csv", encoding='latin-1', sep=',')
liberalCosines = pd.read_csv("wordCosinesliberalsfinal.csv", encoding='latin-1', sep=',')

#data cleaning
del liberalCosines['word']
del conservativeCosines['word']
liberalCosines.rename(columns ={'Unnamed: 0': 'word'}, inplace = True)
conservativeCosines.rename(columns ={'Unnamed: 0': 'word'}, inplace = True) 
liberalCosines.set_index('word', inplace=True)
conservativeCosines.set_index('word', inplace=True)

#--------------------------------------------------------------------------
# Shells for Storing Results 
#--------------------------------------------------------------------------

#lists for storing aggregate yearly data
currentYear = []
propSimilarity = [] #ratio of intersecting words to total words
vectorSimilaritySame = [] #cosine distances betweeen words in common
vectorSimilarityTotal = [] #cosine distances between all words 
freqLibMentions = [] #empty list for number of times lib mentions immigrants or immigration
freqConMentions = [] 
percentLibMentions = [] #empty list for immigration mentions as percentage of all words
percentConMentions = []
totallyUniqueWords = [] #of words in Con or Lib not anywhere in other model
liberalSentiment = [] #valence as measured by dist of immig words (weighted)
#                      by their cosine similarity to a set of seed words
liberalSentimentUnwgt = []
conservativeSentimentUnwgt = []

conservativeSentiment = []
posNegDiffLibConYearList = [] #valence difference of same words
posNegDiffLibConYearListUnwgt = []
posNegDiffLibConYearStdList = [] #valence difference of same words per word

libConDiffPerWordList = []
libConDiffTotalList = []
#--------------------------------------------------------------------------
# Cycle Through Years 
#--------------------------------------------------------------------------

for year in range(1909,2013):
    
    currentYear.append(year)
    label = str(year)
    
    #--------------------------------------------------------------
    #store Index items (words) when coef is not empty for this year
    #--------------------------------------------------------------
    conIndex = conservativeCosines[pd.isnull(conservativeCosines[label]) == False].index.tolist()
    libIndex = liberalCosines[pd.isnull(liberalCosines[label]) == False].index.tolist()
     
    #--------------------------------------------------------------
    #Intersection and Uniques
    #--------------------------------------------------------------
    conOnlyTerms = set(conIndex)-set(libIndex) #terms in con not in lib
    libOnlyTerms = set(libIndex) - set(conIndex)
    jointTerms = list(set(conIndex) & set(libIndex)) #terms in both con and lib
    
    similarityProp = len(jointTerms)/2500 #how many of the words are shared?
    propSimilarity.append(similarityProp)
    
    #--------------------------------------------------------------
    # Loading the pre-trained w2v models
    #--------------------------------------------------------------
    modelConName = 'lipadConservative'+label
    modelLibName = 'lipadLiberal'+label
    modelCon = gensim.models.Word2Vec.load(modelConName)
    modelLib = gensim.models.Word2Vec.load(modelLibName)
    
    
    #--------------------------------------------------------------
    # Sentiment Seeds (From Cochrane et al., 2019)
    #--------------------------------------------------------------
    #Sentiment Seeds for Conservatives
    goodCon = modelCon.wv['good']
    excellentCon = modelCon.wv['excellent']
    correctCon = modelCon.wv['correct']
    bestCon = modelCon.wv['best']
    happyCon = modelCon.wv['happy']
    positiveCon = modelCon.wv['positive']
    fortunateCon = modelCon.wv['fortunate']

    badCon = modelCon.wv['bad']
    terribleCon = modelCon.wv['terrible']
    wrongCon = modelCon.wv['wrong']
    worstCon = modelCon.wv['worst']
    disappointedCon = modelCon.wv['disappointed']
    negativeCon = modelCon.wv['negative']
    unfortunateCon = modelCon.wv['unfortunate']
    
    #Sentiment Seeds for Liberals
    goodLib = modelLib.wv['good']
    excellentLib = modelLib.wv['excellent']
    correctLib = modelLib.wv['correct']
    bestLib = modelLib.wv['best']
    happyLib = modelLib.wv['happy']
    positiveLib = modelLib.wv['positive']
    fortunateLib = modelLib.wv['fortunate']

    badLib = modelLib.wv['bad']
    terribleLib = modelLib.wv['terrible']
    wrongLib = modelLib.wv['wrong']
    worstLib = modelLib.wv['worst']
    disappointedLib = modelLib.wv['disappointed']
    negativeLib = modelLib.wv['negative']
    unfortunateLib = modelLib.wv['unfortunate']
    
    
    
    #------------------------------------------
    # unigrams for immigrants and immigration
    #------------------------------------------
    conMentions = modelCon.wv.vocab["immigrants"].count + modelCon.wv.vocab["immigration"].count
    libMentions = modelLib.wv.vocab["immigrants"].count + modelLib.wv.vocab["immigration"].count
    freqConMentions.append(conMentions)
    freqLibMentions.append(libMentions)
    
    conTokens = modelCon.wv.vectors.shape[0] #the number of words processed
    libTokens = modelLib.wv.vectors.shape[0]
    
    conMentionsPercent = conMentions/conTokens
    libMentionsPercent = libMentions/libTokens
    
    percentConMentions.append(conMentionsPercent)
    percentLibMentions.append(libMentionsPercent)
    
    
    #----------------------
    # Inializing variables 
    #----------------------
    
    libConDiffShared = 0 
    libConDiffTotal = 0

    conSentimentYear = 0 
    libSentimentYear = 0    
    conSentimentYearUnwgt = 0
    libSentimentYearUnwgt = 0
    
    posNegDiffLibConYear = 0
    posNegDiffLibConYearUnwgt = 0
    
    #---------------------------
    # For words shared in common 
    #---------------------------
    
    for word in jointTerms:
        conCoef = conservativeCosines.loc[str(word)][str(year)] #distance to immigrants/immigration
        libCoef = liberalCosines.loc[str(word)][str(year)] #distance to immigrants/immigration
        diff = abs(conCoef - libCoef)
        libConDiffShared += diff
        
        #calculate sentiment of word for Conservatives
        
        word_vector = modelCon.wv[str(word)]
        
        pos1 = np.dot(word_vector, goodCon) / (np.linalg.norm(word_vector) * np.linalg.norm(goodCon))
        pos2 = np.dot(word_vector, excellentCon) / (np.linalg.norm(word_vector) * np.linalg.norm(excellentCon))
        pos3 = np.dot(word_vector, correctCon) / (np.linalg.norm(word_vector) * np.linalg.norm(correctCon))
        pos4 = np.dot(word_vector, bestCon) / (np.linalg.norm(word_vector) * np.linalg.norm(bestCon))
        pos5 = np.dot(word_vector, happyCon) / (np.linalg.norm(word_vector) * np.linalg.norm(happyCon))
        pos6 = np.dot(word_vector, positiveCon) / (np.linalg.norm(word_vector) * np.linalg.norm(positiveCon))
        pos7 = np.dot(word_vector, fortunateCon) / (np.linalg.norm(word_vector) * np.linalg.norm(fortunateCon))
    
        neg1 = np.dot(word_vector, badCon) / (np.linalg.norm(word_vector) * np.linalg.norm(badCon))
        neg2 = np.dot(word_vector, terribleCon) / (np.linalg.norm(word_vector) * np.linalg.norm(terribleCon))
        neg3 = np.dot(word_vector, wrongCon) / (np.linalg.norm(word_vector) * np.linalg.norm(wrongCon))
        neg4 = np.dot(word_vector, worstCon) / (np.linalg.norm(word_vector) * np.linalg.norm(worstCon))
        neg5 = np.dot(word_vector, disappointedCon) / (np.linalg.norm(word_vector) * np.linalg.norm(disappointedCon))
        neg6 = np.dot(word_vector, negativeCon) / (np.linalg.norm(word_vector) * np.linalg.norm(negativeCon))
        neg7 = np.dot(word_vector, unfortunateCon) / (np.linalg.norm(word_vector) * np.linalg.norm(unfortunateCon))
     
        pos = sum([pos1, pos2, pos3, pos4, pos5, pos6, pos7])
        neg = sum([neg1, neg2, neg3, neg4, neg5, neg6, neg7])
        
        if conCoef > 0:
            posNegCon = (pos-neg)*conCoef #weighted
            posNegConUnwgt = pos-neg
        else:
            posNegCon = np.nan
            posNegConUnwgt = np.nan
         #weighted
        
        conSentimentYear += posNegCon
        conSentimentYearUnwgt += posNegConUnwgt
        
        #calculate sentiment of word for Conservatives
        
        word_vector = modelLib.wv[str(word)]
        
        pos1 = np.dot(word_vector, goodLib) / (np.linalg.norm(word_vector) * np.linalg.norm(goodLib))
        pos2 = np.dot(word_vector, excellentLib) / (np.linalg.norm(word_vector) * np.linalg.norm(excellentLib))
        pos3 = np.dot(word_vector, correctLib) / (np.linalg.norm(word_vector) * np.linalg.norm(correctLib))
        pos4 = np.dot(word_vector, bestLib) / (np.linalg.norm(word_vector) * np.linalg.norm(bestLib))
        pos5 = np.dot(word_vector, happyLib) / (np.linalg.norm(word_vector) * np.linalg.norm(happyLib))
        pos6 = np.dot(word_vector, positiveLib) / (np.linalg.norm(word_vector) * np.linalg.norm(positiveLib))
        pos7 = np.dot(word_vector, fortunateLib) / (np.linalg.norm(word_vector) * np.linalg.norm(fortunateLib))
    
        neg1 = np.dot(word_vector, badLib) / (np.linalg.norm(word_vector) * np.linalg.norm(badLib))
        neg2 = np.dot(word_vector, terribleLib) / (np.linalg.norm(word_vector) * np.linalg.norm(terribleLib))
        neg3 = np.dot(word_vector, wrongLib) / (np.linalg.norm(word_vector) * np.linalg.norm(wrongLib))
        neg4 = np.dot(word_vector, worstLib) / (np.linalg.norm(word_vector) * np.linalg.norm(worstLib))
        neg5 = np.dot(word_vector, disappointedLib) / (np.linalg.norm(word_vector) * np.linalg.norm(disappointedLib))
        neg6 = np.dot(word_vector, negativeLib) / (np.linalg.norm(word_vector) * np.linalg.norm(negativeLib))
        neg7 = np.dot(word_vector, unfortunateLib) / (np.linalg.norm(word_vector) * np.linalg.norm(unfortunateLib))
     
        pos = sum([pos1, pos2, pos3, pos4, pos5, pos6, pos7])
        neg = sum([neg1, neg2, neg3, neg4, neg5, neg6, neg7])
        
        
        if libCoef > 0:
            posNegLib = (pos-neg) *libCoef #weighted
            posNegLibUnwgt = pos-neg
        else:
            posNegLib = np.nan
            posNegLibUnwgt = np.nan
        
        
        libSentimentYear += posNegLib
        libSentimentYearUnwgt += posNegLibUnwgt
        
        posNegDiffLibCon = abs(posNegLib-posNegCon)
        posNegDiffLibConUnwgt = abs(posNegLibUnwgt - posNegConUnwgt)
        
        posNegDiffLibConYear += posNegDiffLibCon
        posNegDiffLibConYearUnwgt += posNegDiffLibConUnwgt

    
    posNegDiffLibConYearList.append(posNegDiffLibConYear)
    posNegDiffLibConYearListUnwgt.append(posNegDiffLibConUnwgt)
    
    posNegDiffLibConYearStd = posNegDiffLibConYear/len(jointTerms)
    posNegDiffLibConYearStdList.append(posNegDiffLibConYearStd)
    
    
    libConDiffPerWordList.append(libConDiffShared/len(jointTerms))
    libConDiffTotalList.append(libConDiffShared)
    
    totalUniques = 0
    
    for word in libOnlyTerms:
        libCoef = liberalCosines.loc[str(word)][str(year)]
        try: #to find word coef in Conservative data

            immigrants = modelCon.wv['immigrants']
            immigration = modelCon.wv['immigration']          
            word_model = modelCon.wv[word] 
            immig1 = np.dot(word_model, immigrants) / (np.linalg.norm(word_model) * np.linalg.norm(immigrants))
            immig2 = np.dot(word_model, immigration) / (np.linalg.norm(word_model) * np.linalg.norm(immigration))   
            conCoef = sum([immig1, immig2])
            
            libConDiffTotal = libConDiffTotal + abs(libCoef - conCoef)
            

            
        
        except: #if not found
            #print(word, " not found in Conservative Model")
            libConDiffTotal = libConDiffTotal + abs(libCoef) #treat Con as 0 and add value of lib coefficient to total diff
            totalUniques +=1
        
        #Get Sentiment of the Word for Liberalsd
        
                
        word_vector = modelLib.wv[str(word)]
        
        pos1 = np.dot(word_vector, goodLib) / (np.linalg.norm(word_vector) * np.linalg.norm(goodLib))
        pos2 = np.dot(word_vector, excellentLib) / (np.linalg.norm(word_vector) * np.linalg.norm(excellentLib))
        pos3 = np.dot(word_vector, correctLib) / (np.linalg.norm(word_vector) * np.linalg.norm(correctLib))
        pos4 = np.dot(word_vector, bestLib) / (np.linalg.norm(word_vector) * np.linalg.norm(bestLib))
        pos5 = np.dot(word_vector, happyLib) / (np.linalg.norm(word_vector) * np.linalg.norm(happyLib))
        pos6 = np.dot(word_vector, positiveLib) / (np.linalg.norm(word_vector) * np.linalg.norm(positiveLib))
        pos7 = np.dot(word_vector, fortunateLib) / (np.linalg.norm(word_vector) * np.linalg.norm(fortunateLib))
    
        neg1 = np.dot(word_vector, badLib) / (np.linalg.norm(word_vector) * np.linalg.norm(badLib))
        neg2 = np.dot(word_vector, terribleLib) / (np.linalg.norm(word_vector) * np.linalg.norm(terribleLib))
        neg3 = np.dot(word_vector, wrongLib) / (np.linalg.norm(word_vector) * np.linalg.norm(wrongLib))
        neg4 = np.dot(word_vector, worstLib) / (np.linalg.norm(word_vector) * np.linalg.norm(worstLib))
        neg5 = np.dot(word_vector, disappointedLib) / (np.linalg.norm(word_vector) * np.linalg.norm(disappointedLib))
        neg6 = np.dot(word_vector, negativeLib) / (np.linalg.norm(word_vector) * np.linalg.norm(negativeLib))
        neg7 = np.dot(word_vector, unfortunateLib) / (np.linalg.norm(word_vector) * np.linalg.norm(unfortunateLib))
     
        pos = sum([pos1, pos2, pos3, pos4, pos5, pos6, pos7])
        neg = sum([neg1, neg2, neg3, neg4, neg5, neg6, neg7])
        
        if libCoef > 0:
            posNegLib = (pos-neg) *libCoef #weighted
            posNegLibUnwgt = pos-neg
        else:
            posNegLib = np.nan
            posNegLibUnwgt = np.nan
        
        libSentimentYear += posNegLib
        libSentimentYearUnwgt += posNegLibUnwgt
        
        
              
    for word in conOnlyTerms:
        conCoef = conservativeCosines.loc[str(word)][str(year)]
        try: #to find word coef in Liberal data
            word_model = modelLib.wv[word] 
            
            immigrants = modelLib.wv['immigrants']
            immigration = modelLib.wv['immigration']          
            
            immig1 = np.dot(word_model, immigrants) / (np.linalg.norm(word_model) * np.linalg.norm(immigrants))
            immig2 = np.dot(word_model, immigration) / (np.linalg.norm(word_model) * np.linalg.norm(immigration))   
            libCoef = sum([immig1, immig2]) #ERROR CAUGHT ON 2019-04-15 Originally labelled "concoef"
            
            libConDiffTotal = libConDiffTotal + abs(libCoef - conCoef)
        
        except: #if not found
            #3print(word, " not found in Liberal Model")
            libConDiffTotal = libConDiffTotal + abs(conCoef) #treat Lib as 0 and add value of Con coefficient to total diff
            totalUniques+=1
        
        
        #Get sentiment of the word for Conservatives
        
        word_vector = modelCon.wv[str(word)]
        
        pos1 = np.dot(word_vector, goodCon) / (np.linalg.norm(word_vector) * np.linalg.norm(goodCon))
        pos2 = np.dot(word_vector, excellentCon) / (np.linalg.norm(word_vector) * np.linalg.norm(excellentCon))
        pos3 = np.dot(word_vector, correctCon) / (np.linalg.norm(word_vector) * np.linalg.norm(correctCon))
        pos4 = np.dot(word_vector, bestCon) / (np.linalg.norm(word_vector) * np.linalg.norm(bestCon))
        pos5 = np.dot(word_vector, happyCon) / (np.linalg.norm(word_vector) * np.linalg.norm(happyCon))
        pos6 = np.dot(word_vector, positiveCon) / (np.linalg.norm(word_vector) * np.linalg.norm(positiveCon))
        pos7 = np.dot(word_vector, fortunateCon) / (np.linalg.norm(word_vector) * np.linalg.norm(fortunateCon))
    
        neg1 = np.dot(word_vector, badCon) / (np.linalg.norm(word_vector) * np.linalg.norm(badCon))
        neg2 = np.dot(word_vector, terribleCon) / (np.linalg.norm(word_vector) * np.linalg.norm(terribleCon))
        neg3 = np.dot(word_vector, wrongCon) / (np.linalg.norm(word_vector) * np.linalg.norm(wrongCon))
        neg4 = np.dot(word_vector, worstCon) / (np.linalg.norm(word_vector) * np.linalg.norm(worstCon))
        neg5 = np.dot(word_vector, disappointedCon) / (np.linalg.norm(word_vector) * np.linalg.norm(disappointedCon))
        neg6 = np.dot(word_vector, negativeCon) / (np.linalg.norm(word_vector) * np.linalg.norm(negativeCon))
        neg7 = np.dot(word_vector, unfortunateCon) / (np.linalg.norm(word_vector) * np.linalg.norm(unfortunateCon))
     
        pos = sum([pos1, pos2, pos3, pos4, pos5, pos6, pos7])
        neg = sum([neg1, neg2, neg3, neg4, neg5, neg6, neg7])
        
        if conCoef > 0:
            posNegCon = (pos-neg)*conCoef #weighted
            posNegConUnwgt = pos-neg
        else:
            posNegCon = np.nan
            posNegConUnwgt = np.nan
         #weighted
        
        conSentimentYear += posNegCon
        conSentimentYearUnwgt += posNegConUnwgt
          
    vectorSimilaritySame.append(libConDiffShared) #cosine distances betweeen words in Common
    vectorSimilarityTotal.append(libConDiffTotal)
    totallyUniqueWords.append(totalUniques)
    conservativeSentiment.append(conSentimentYear)
    liberalSentiment.append(libSentimentYear)
    conservativeSentimentUnwgt.append(conSentimentYearUnwgt)
    liberalSentimentUnwgt.append(libSentimentYearUnwgt)
    
    
    
    
    
    print("finished ", year)
    


#output to dataFrame    

yearlyData = pd.DataFrame(
        {'Year': currentYear,
         'SimilarityProportion': propSimilarity,
         'freqConMentions': freqConMentions,
         'freqLibMentions': freqLibMentions,
         'percentConMentions': percentConMentions,
         'percentLibMentions': percentLibMentions,
         'cosineDistanceCommonTerms': vectorSimilaritySame,
         'cosineDistancePerSharedWord': libConDiffPerWordList,
         'cosineDistanceAllTerms': libConDiffTotalList,
         'totallyUniqueWords': totallyUniqueWords,
         'liberalSentiment': liberalSentiment,
         'conservativeSentiment': conservativeSentiment,
         'liberalSentimentUnwgt': liberalSentimentUnwgt,
         'conservativeSentimentUnwgt': conservativeSentimentUnwgt,
         'valenceDifferenceSameWords': posNegDiffLibConYearList,
         'valenceDifferencesSameWordsStd': posNegDiffLibConYearStdList,
         'valenceDifferenceSameWordsUnwgt': posNegDiffLibConYearListUnwgt}) 
    
yearlyData.to_csv('yearlyDataFull.csv')    



