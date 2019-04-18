# C. Cochrane, University of Toronto, 2019-04-17
#-----------------------------------------------------------------
# Description
#-----------------------------------------------------------------

# A script for generating Figure 1: Conceputalizing Consensus

#-----------------------------------------------------------------
# Requirements
#-----------------------------------------------------------------

# (1) R-Studio; (2) rstudioapi; (3) tidyverse

#-----------------------------------------------------------------
# Initialization
#-----------------------------------------------------------------

library(tidyverse)
library(rstudioapi)

#set working directory to folder containing script 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

#-----------------------------------------------------------------
# Build Graph
#-----------------------------------------------------------------

# Define labels with subscripts
label1 <- "S[1]"
label2 <- "S[2]"
label3 <- "S[3]"
label4 <- "S[4]"

# Generate empty data frame

df = data.frame()

# Graph
graph <- ggplot(df) +
  geom_hline(yintercept =5) + 
  geom_vline(xintercept =5) + 
  scale_x_continuous(
                     limits=c(0,10), 
                     breaks=c(2.5,7.5), 
                     labels=c('Disagree', 'Agree')
                     ) + 
  scale_y_continuous(
                     limits=c(0,10), 
                     breaks=c(3.5, 8.5), 
                     labels=c('Disagree', 'Agree')
                     ) +
  theme(panel.background=element_rect(fill="white")) +
  labs(x="\n Valence", y="Conceptualization\n") +
  annotate("text", x=7.5, y=7.5, label = label1, parse=TRUE, size=15) +
  annotate("text", x=2.5, y=7.5, label = label3, parse=TRUE, size=15) +
  annotate("text", x=2.5, y=2.5, label = label4, parse=TRUE, size=15) +
  annotate("text", x=7.5, y=2.5, label = label2, parse=TRUE, size=15) +
  geom_segment(
               mapping=aes(x=2.75, y=2.75, xend=7, yend=7), 
               arrow=arrow(), 
               color="blue4", 
               size=2
               ) +
  geom_segment(
              mapping=aes(x=2.75, y=2.75, xend=2.75, yend=7), 
              arrow=arrow(), 
              color="blue4", 
              size=1
              ) +
  geom_segment(
            mapping=aes(x=2.75, y=2.75, xend=7, yend=2.75), 
            arrow=arrow(), 
            color="blue4", 
            size=1
            ) +
  geom_segment(
            mapping=aes(x=3, y=7.5, xend=7, yend=7.5), 
            arrow=arrow(), 
            color="blue4", 
            size=1
            ) +
  geom_segment(
            mapping=aes(x=7.5, y=3, xend=7.5, yend=6.75), 
            arrow=arrow(), 
            color="blue4", 
            size=1
            ) +
  theme(
        axis.text.x = element_text(size=15),
        axis.text.y = element_text(size=15, angle=90),
        axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        legend.text = element_text(size=5),
        legend.title = element_blank()
        )

graph

ggsave("consensus.pdf", device="pdf", width=10, height=10)


