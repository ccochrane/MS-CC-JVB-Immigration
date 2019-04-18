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
library(tidyverse)
library(ggrepel)

library(rstudioapi)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#-----------------------------------------------------------------
# Load Data
#-----------------------------------------------------------------

wordData <- read.csv('wordData.csv')

#-----------------------------------------------------------------
# Graph Liberal and Conservative Sentiment (Common words)
#-----------------------------------------------------------------

wordDataJoint <-
  wordData %>% filter(conUnique == "no" & libUnique=="no")

#associated with immigration among Liberals and Conservatives
wordDataJointLTD <-
  wordDataJoint %>% filter(conCoef > .5 & libCoef>.5)

#extracting words with maximum difference
wordDataJointLTD$diffSentABS <- abs(wordDataJointLTD$diffSentUnwgt) 

wordJointHighDiffCon <- top_n(wordDataJointLTD, 50, diffSentUnwgt) #most diff +CON
wordJointHighDiffLib <- top_n(wordDataJointLTD, -50, diffSentUnwgt) #most diff +LIB

wordDataJointLTD$meanSent <- (wordDataJointLTD$conSent+wordDataJointLTD$libSent)/2

wordJointHighSent <- top_n(wordDataJointLTD, 50, meanSent)
wordJointLowSent <- top_n(wordDataJointLTD, -50, meanSent)

wordJointDiffs <- bind_rows(wordJointHighDiffCon, 
                            wordJointHighDiffLib,
                            wordJointHighSent,
                            wordJointLowSent)

graphDiffs <- ggplot() +
  geom_text_repel(data=wordJointDiffs, 
            aes(x=libSentUnwgt, 
                y=conSentUnwgt, 
                label=wordYear,), stat="identity", color="black") +
  geom_point(data=wordDataJointLTD, 
             aes(x=libSentUnwgt, 
                 y=conSentUnwgt), color="grey10", shape=1, alpha=.2) +
  theme(axis.text.x = element_text(size=15)) +
  theme(axis.text.y = element_text(size=15)) +
  theme(panel.background=element_blank()) +
  theme(panel.grid.minor=element_line(color="grey")) + 
  theme(axis.title.x = element_text(size=20)) +
  theme(axis.title.y = element_text(size=20)) +
  ylab('Conservative Valence\n ') +
  xlab('\n Liberal Valence') +
  scale_x_continuous(limits=c(-2.25,2.25)) +
  scale_y_continuous(limits=c(-2.25,2.25))



graphDiffs
ggsave("sentDiffsUnwgt.pdf", device="pdf", width=20, height=10)


#-----------------------------------------------------------------
# Graph Liberal and Conservative Sentiment by Year
#-----------------------------------------------------------------

wordData$LibConSentDiff = abs(wordData$conSent - wordData$libSent)
wordData %<>%
  group_by(Year) %>%
  summarize(averageSentDiff=mean(LibConSentDiff, na.rm=T))

wordSentSimilarityGraph <- ggplot(data=wordData, aes(x=Year))+ 
  geom_line(aes(y=averageSentDiff), color="black") +
  geom_smooth(aes(y=averageSentDiff), color="blue", se=FALSE, span=.2) +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_text(angle=90, size=10)) +
  scale_x_continuous(breaks=seq(1908,2013,3)) +
  ylab("Sentiment Differences Per Shared Word\n") +
  theme(axis.title.y = element_text(size=10)) +
  theme(axis.text.x = element_text(size=10))


wordSentSimilarityGraph
#ggsave('wordSentSimilarity.pdf', device='pdf', height=8, width=12)



fit <- lm(libSentUnwgt ~ conSentUnwgt + Year, data=wordData)
summary(fit)


