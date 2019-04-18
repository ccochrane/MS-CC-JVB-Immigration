# MS-CC-JVB-Immigration
An analysis of parliamentary discourse re: immigrants/immigration

(1) queryScript.py - script for extracting LiPaD data from MySQL.
(2) yearByYearbyPartyModelRunConservative.py - script for training word2vec on 11-year moving subsets of Conservative speeches in LiPaD data.
(3) coefficientsToDataFrameConservatives.py - extracts the 2500 words most closely associated with immigration/immigrants among Conservatives and stores the word, year, coefficient groups as a Data Frame.
(4) yearByYearbyPartyModelRunLiberal.py - script for training word2vec on 11-year moving subsets of Liberal speeches in LiPaD data.
(5) coefficientsToDataFrameLiberals.py - extracts the 2500 words most closely associated with immigration/immigrants among Liberals and stores the word, year, coefficient groups as a Data Frame.
(6) yearByYearLevelAnalysis.py - script for analyzing at the year level the vector representations of words/bi-grams in the LiPaD corpus generated from Steps (2) - (5).
(7) wordByWordLevelAnalysis.py - script for analyzing at the word level the vector representations of words generated from Steps (2) - (5).
(8) yearByYearGraphs.R - script for generating graphs to summarize the analysis generated in Step (6).
(9) conLibWordValenceGraph.py - script for generating graphs to summarize the word-level analysis generated in Step (7).
(10) uniqueWords.R - script for generating graphs to summarize the word-level properties of unique words, generated in Step (7).



