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
# Graph Unique Words (Conservative)
#-----------------------------------------------------------------

wordDataConUniques <-
  wordData %>% filter(conUnique == "yes" & libUnique=="no")

#downweight words not at all associated with immigration:
wordDataConUniques$conCoef01[wordDataConUniques$conCoef<0] <- 0 
wordDataConUniques$conCoef01[wordDataConUniques$conCoef>=0] <- 1
#if word associted with immigration, multiply its sentiment
#by its cosine similarity to immigration
wordDataConUniques$conCoefSent <- (wordDataConUniques$conCoef01*
                                   wordDataConUniques$conCoef)*
                                   wordDataConUniques$conSentUnwgt
#absolute value of sentiment
wordDataConUniques$conCoefSentAbs <- abs(wordDataConUniques$conCoefSent)

wordConUniquesHighSalience <- top_n(
                                    wordDataConUniques,    
                                    50, 
                                    conCoefSentAbs
                                    ) #top 50 most important



wordDataOutliers <- wordConUniquesHighSalience

conUniquesGraph <- ggplot() +
  geom_text_repel(
                  data=wordDataOutliers, 
                  aes(
                      x=conSentUnwgt, 
                      y=conCoef, 
                      label=wordYear
                      ), 
                  stat="identity", 
                  color="black"
                  ) +
  geom_point(
             data=wordDataConUniques, 
             aes(
                 x=conSentUnwgt, 
                 y=conCoef,
                 color=conCoefSentAbs
                 ), 
             shape=1, 
             alpha=.2
             ) +
  theme(axis.text.x = element_text(size=15)) +
  theme(axis.text.y = element_text(size=15)) +
  theme(panel.background=element_blank()) +
  theme(panel.grid.minor=element_line(color="grey")) + 
  theme(axis.title.x = element_text(size=20)) +
  theme(axis.title.y = element_text(size=20)) +
  ylab('Cosine Similarity to Immigration\n ') +
  xlab('\n Valence') +
  scale_x_continuous(limits=c(-2.5,2.5)) +
  scale_y_continuous(limits=c(0,1.5)) +
  scale_color_continuous(low="darkgreen", high="red", limits=c(0,.75))+
  labs(color="Salience")



conUniquesGraph
ggsave("conUniques.pdf", device="pdf", width=20, height=10)


#-----------------------------------------------------------------
# Graph Unique Words (Liberals)
#-----------------------------------------------------------------

wordDataLibUniques <-
  wordData %>% filter(conUnique == "no" & libUnique=="yes")

#wordDataConUniquesLTD <-
#  wordDataConUniques %>% filter(conCoef > .5)

wordDataLibUniques$libCoef01[wordDataLibUniques$libCoef<0] <- 0
wordDataLibUniques$libCoef01[wordDataLibUniques$libCoef>=0] <- 1
wordDataLibUniques$libCoefSent <- (wordDataLibUniques$libCoef01*
                                     wordDataLibUniques$libCoef)*
                                     wordDataLibUniques$libSentUnwgt

wordDataLibUniques$libCoefSentAbs <- abs(wordDataLibUniques$libCoefSent)

wordLibUniquesHighSalience <- top_n(wordDataLibUniques, 50, libCoefSentAbs) #most important



wordDataOutliers <- wordLibUniquesHighSalience

libUniquesGraph <- ggplot() +
  geom_text_repel(
                  data=wordDataOutliers, 
                  aes(
                      x=libSentUnwgt, 
                      y=libCoef, 
                      label=wordYear
                      ), 
                  stat="identity", 
                  color="black"
                  ) +
  geom_point(
             data=wordDataLibUniques, 
             aes(
                 x=libSentUnwgt, 
                 y=libCoef,
                 color=libCoefSentAbs
                 ), 
             shape=1, 
             alpha=.2) +
  theme(axis.text.x = element_text(size=15)) +
  theme(axis.text.y = element_text(size=15)) +
  theme(panel.background=element_blank()) +
  theme(panel.grid.minor=element_line(color="grey")) + 
  theme(axis.title.x = element_text(size=20)) +
  theme(axis.title.y = element_text(size=20)) +
  ylab('Cosine Similarity to Immigration\n ') +
  xlab('\n Valence') +
  scale_x_continuous(limits=c(-2.5,2.5)) +
  scale_y_continuous(limits=c(0,1.5)) +
  scale_color_continuous(low="darkgreen", high="red", limits=c(0,.75))+
  labs(color="Salience")



libUniquesGraph
ggsave("libUniques.pdf", device="pdf", width=20, height=10)
