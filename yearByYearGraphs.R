# C. Cochrane, University of Toronto, 2019-04-17
#-----------------------------------------------------------------
# Description
#-----------------------------------------------------------------

# A script for Graphing Liberal and Conservative Valence
# of words associated with immigration.

#-----------------------------------------------------------------
# Requirements
#-----------------------------------------------------------------

# (1) R-Studio; (2) rstudioapi; (3) tidyverse

#-----------------------------------------------------------------
# Initialization
#-----------------------------------------------------------------
library(ggplot2)

library(rstudioapi)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
sentimentData <- read.csv('immigrationSentimentByParty.csv')

#-----------------------------------------------------------------
# Load Data
#-----------------------------------------------------------------

yearlyData <- read.csv('yearlyDataFull.csv')

#-----------------------------------------------------------------
# Graph Yearly Sentiment
#-----------------------------------------------------------------

sentimentPlot <- ggplot(data=yearlyData, aes(x=Year))+
  geom_line(aes(y=conservativeSentimentUnwgt), color="blue") +
  geom_line(aes(y=liberalSentimentUnwgt), color="red") +
  geom_smooth(aes(y=conservativeSentiment), color="blue", se=FALSE, span=.2) +
  geom_smooth(aes(y=liberalSentiment), color="red", se=FALSE, span=.2) +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_text(angle=90, size=10)) +
  scale_x_continuous(breaks=seq(1908,2013,3)) +
  ylab("Sentiment") +
  theme(axis.title.y = element_text(size=10)) +
  theme(axis.text.x = element_text(size=10))

sentimentPlot
ggsave('sentimentPlot.pdf', device='pdf', height=8, width=12)


#-----------------------------------------------------------------
# Graph Cosine Distance PerShared Word
#-----------------------------------------------------------------

#Note -- this is the sum of the absolute value of the differences, 
#for each word, of its cosine distance to "immigrants"/"immigration"
#among Conservatives and Liberals. 


cosSimilarityPlot <- ggplot(data=yearlyData, aes(x=Year))+ 
  geom_line(aes(y=cosineDistancePerSharedWord), color="black") +
  geom_smooth(aes(y=cosineDistancePerSharedWord), color="blue", se=FALSE, span=.2) +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_text(angle=90, size=10)) +
  scale_x_continuous(breaks=seq(1908,2013,3)) +
  ylab("Cosine Differences Per Shared Word\n") +
  theme(axis.title.y = element_text(size=10)) +
  theme(axis.text.x = element_text(size=10))


cosSimilarityPlot
ggsave('cosSimilarityPlot.pdf', device='pdf', height=8, width=12)


#-----------------------------------------------------------------
# Graph Yearly Number of Totally Unique Words in each year
#-----------------------------------------------------------------

#A totally unique word is a word associated by one party with
#immigration that appears nowhere in the vocabulary of the other
#party

totalUniquesFreq <- ggplot(data=yearlyData, aes(x=Year))+ 
  geom_line(aes(y=totallyUniqueWords), color="black") +
  geom_smooth(aes(y=totallyUniqueWords), color="blue", se=FALSE, span=.2) +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_text(angle=90, size=10)) +
  scale_x_continuous(breaks=seq(1908,2013,3)) +
  ylab("Number of Totally Unique Words") +
  theme(axis.title.y = element_text(size=10)) +
  theme(axis.text.x = element_text(size=10))

totalUniquesFreq
ggsave('totalUniquesPlot.pdf', device='pdf', height=8, width=12)

#-----------------------------------------------------------------
# Graph the n-grams "immigration" and "immigrants" for each year.
#-----------------------------------------------------------------

attentionPlot <- ggplot(data=yearlyData, aes(x=Year))+
  geom_line(aes(y=percentConMentions), color="blue") +
  geom_line(aes(y=percentLibMentions), color="red") +
  geom_smooth(aes(y=percentConMentions), color="blue", se=FALSE, span=.2) +
  geom_smooth(aes(y=percentLibMentions), color="red", se=FALSE, span=.2) +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_text(angle=90, size=10)) +
  scale_x_continuous(breaks=seq(1908,2013,3)) +
  ylab("% Immig[] Among Total Words") +
  theme(axis.title.y = element_text(size=10)) +
  theme(axis.text.x = element_text(size=10))

attentionPlot

ggsave('attentionPlot.pdf', device='pdf', height=8, width=12)

