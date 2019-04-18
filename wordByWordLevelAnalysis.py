# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:04:34 2019

@author: chris
"""

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

conservativeCosines = pd.read_csv("wordCosinesConservatives.csv", encoding='latin-1', sep=',')
liberalCosines = pd.read_csv("wordCosinesliberalsfinal.csv", encoding='latin-1', sep=',')


#Data Cleaning
del liberalCosines['word']
del conservativeCosines['word']
liberalCosines.rename(columns ={'Unnamed: 0': 'word'}, inplace = True)
conservativeCosines.rename(columns ={'Unnamed: 0': 'word'}, inplace = True)
liberalCosines.set_index('word', inplace=True)
conservativeCosines.set_index('word', inplace=True)




#Convert the indexes to lists
conservativeIndex = conservativeCosines.index.tolist()
liberalIndex = liberalCosines.index.tolist()
jointTerms = set(conservativeIndex + liberalIndex)
conOnlyTerms = set(conservativeIndex)-set(liberalIndex)
libOnlyTerms = set(liberalIndex) - set(conservativeIndex)



#lists for aggregate yearly data
yearList = []
wordYearList = []
currentYearList = []
diffCoefList = [] #diff Coef, con + and lib -
diffSentList = [] #diff Sent, con + and lib -
diffSentListUnwgt = [] #diff Sent, not weighted by coef
libCoefList = []
conCoefList = []
libSentList = []
libSentListUnwgt = []
conSentList = []
conSentListUnwgt = []
#Lists for aggregate party data
conUniqueList = []
libUniqueList = []


for year in range(1909,2013):
    label = str(year)
    #extracts the top 2500 words most closely linked to immigration for each party in each year
    conIndex = conservativeCosines[pd.isnull(conservativeCosines[label]) == False].index.tolist()
    libIndex = liberalCosines[pd.isnull(liberalCosines[label]) == False].index.tolist()
    
    #Intersection and Uniques
    conOnlyTerms = set(conIndex)-set(libIndex)
    libOnlyTerms = set(libIndex) - set(conIndex)
    jointTerms = list(set(conIndex) & set(libIndex))

    #Loading the Models
    modelConName = 'lipadConservative'+label
    modelLibName = 'lipadLiberal'+label
    modelCon = gensim.models.Word2Vec.load(modelConName)
    modelLib = gensim.models.Word2Vec.load(modelLibName)
    
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
    
    for word in jointTerms:
        yearList.append(year)
        wordYear = (word, year)
        wordYearList.append(wordYear)
        
        libUniqueList.append("no") #not unique, in joint terms
        conUniqueList.append("no")
        
        conCoef = conservativeCosines.loc[str(word)][str(year)] #distance to immigrants/immigration
        libCoef = liberalCosines.loc[str(word)][str(year)] #distance to immigrants/immigration
        
        conCoefList.append(conCoef)
        libCoefList.append(libCoef)
        diffCoef = conCoef - libCoef
        diffCoefList.append(diffCoef) #add to diff Coef
        
   
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
        
        #want to compare libs/con sentiment for words both associate with immig
        if conCoef > 0:
            posNegCon = (pos-neg) *conCoef
            posNegConUnwgt = (pos-neg)
        else:
            posNegCon = np.nan
            posNegConUnw = np.nan
        
        conSentList.append(posNegCon)
        conSentListUnwgt.append(posNegConUnwgt)
        

        #calculate sentiment of word for Liberals
        
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
            posNegLib = (pos-neg)*libCoef
            posNegLibUnwgt = pos-neg
        else:
            posNegLib = np.nan
            posNegLibUnwgt = np.nan
        
        
        libSentList.append(posNegLib)
        libSentListUnwgt.append(posNegLibUnwgt)

        
        libConDiffSent = posNegCon - posNegLib
        libConDiffSentUnwgt = posNegConUnwgt - posNegLibUnwgt
        
        diffSentList.append(libConDiffSent)
        diffSentListUnwgt.append(libConDiffSentUnwgt)



    totalUniques = 0
    
    for word in libOnlyTerms:
        
        yearList.append(year)
        wordYear = (word, year)
        wordYearList.append(wordYear)
       
        
        libCoef = liberalCosines.loc[str(word)][str(year)] #distance to immigrants/immigration
       
        
        libCoefList.append(libCoef)
        
        
        #Get the Liberal Sentiment
        
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
            
        libSentList.append(posNegLib)
        libSentListUnwgt.append(posNegLibUnwgt)
        
        
        
        
        try: #to find word coef in Conservative data

            immigrants = modelCon.wv['immigrants']
            immigration = modelCon.wv['immigration']          
            word_model = modelCon.wv[word] 
            immig1 = np.dot(word_model, immigrants) / (np.linalg.norm(word_model) * np.linalg.norm(immigrants))
            immig2 = np.dot(word_model, immigration) / (np.linalg.norm(word_model) * np.linalg.norm(immigration))   
            conCoef = sum([immig1, immig2])
            conCoefList.append(conCoef)
            
            diffCoef = conCoef - libCoef
            diffCoefList.append(diffCoef) #add to diff Coef
            
            #Get Conservative Sentiment
            
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
                posNegCon = (pos-neg) * conCoef #weighted
                posNegConUnwgt = pos-neg
            else:
                posNegCon = np.nan
                posNegConUnwgt = np.nan
                
            conSentList.append(posNegCon)
            conSentListUnwgt.append(posNegConUnwgt)
            
            libConDiffSent = posNegCon - posNegLib
            libConDiffSentUnwgt = posNegConUnwgt - posNegLibUnwgt
            
            diffSentList.append(libConDiffSent)
            diffSentListUnwgt.append(libConDiffSentUnwgt)
            
                        
            libUniqueList.append("no") #not unique, as word was found in con list
            conUniqueList.append("no")   

        except: #if not found
            #print(word, " not found in Conservative Model")
            
            diffCoef = -libCoef #given no conservative coef
            diffCoefList.append(diffCoef)
            
            conCoefList.append(np.NaN) #no ConCoef
            conSentList.append(np.NaN) #no ConSent
            conSentListUnwgt.append(np.NaN)
            
            libConDiffSent = -posNegLib # as no word found on Con side
            libConDiffSentUnwgt = -posNegLibUnwgt
            diffSentList.append(libConDiffSent)
            diffSentListUnwgt.append(libConDiffSentUnwgt)
            
            libUniqueList.append("yes") #unique, word not in Conservative list
            conUniqueList.append("no")            
            

        
        #Get Sentiment of the Word for Liberalsd
        
                
        
        
              
    for word in conOnlyTerms:
        
        yearList.append(year)
        wordYear = (word, year)
        wordYearList.append(wordYear)
        
        conCoef = conservativeCosines.loc[str(word)][str(year)]
        
        conCoefList.append(conCoef)
        
        #find Con Sentiment
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
        
        conSentList.append(posNegCon)
        conSentListUnwgt.append(posNegConUnwgt)
    
    
        try: #to find word coef in Liberal data
            immigrants = modelLib.wv['immigrants']
            immigration = modelLib.wv['immigration']          
            word_model = modelLib.wv[word] 
            immig1 = np.dot(word_model, immigrants) / (np.linalg.norm(word_model) * np.linalg.norm(immigrants))
            immig2 = np.dot(word_model, immigration) / (np.linalg.norm(word_model) * np.linalg.norm(immigration))   
            libCoef = sum([immig1, immig2])
            
            libCoefList.append(libCoef)
            
            diffCoef = conCoef - libCoef
            diffCoefList.append(diffCoef)
            
            #Get Liberal Sentiment
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
            
            libSentList.append(posNegLib)
            libSentListUnwgt.append(posNegLibUnwgt)
            
            libConDiffSent = posNegCon - posNegLib
            libConDiffSentUnwgt = posNegConUnwgt - posNegLibUnwgt
            
            diffSentList.append(libConDiffSent)
            diffSentListUnwgt.append(libConDiffSentUnwgt)
            
            libUniqueList.append("no") #not unique, as word was found in con list
            conUniqueList.append("no")  
        
        
        except: #if not found

         
            totalUniques+=1
            
            libCoefList.append(np.NaN) #no Lib Coef
            libSentList.append(np.NaN) #no Lib Sent
            libSentListUnwgt.append(np.NaN)
            
            diffCoef = conCoef #given no liberal coef
            diffCoefList.append(diffCoef)
            
            
            libConDiffSent = posNegCon # as no word found on lib side
            libConDiffSentUnwgt = posNegConUnwgt
            diffSentList.append(libConDiffSent)
            diffSentListUnwgt.append(libConDiffSentUnwgt)
            
            libUniqueList.append("no") #not unique, as word not found in list all
            conUniqueList.append("yes")  
        
        
        #Get sentiment of the word for Conservatives
        
    
    print("finished ", year)
    

print(len(yearList))
print(len(wordYearList))
print(len(diffCoefList))
print(len(diffSentList))
print(len(libCoefList))
print(len(libSentList))
print(len(conCoefList))
print(len(conSentList))
print(len(conUniqueList))
print(len(libUniqueList))
print(len(diffSentListUnwgt))

wordData = pd.DataFrame(
        {'Year': yearList,         
         'wordYear': wordYearList,
         'diffCoef': diffCoefList,
         'diffSent': diffSentList,
         'diffSentUnwgt': diffSentListUnwgt,
         'libCoef': libCoefList,
         'libSent': libSentList,
         'libSentUnwgt': libSentListUnwgt,
         'conCoef': conCoefList,
         'conSent': conSentList,
         'conSentUnwgt': conSentListUnwgt,
         'conUnique': conUniqueList,
         'libUnique': libUniqueList}) 
    
wordData.to_csv('wordData.csv')    


