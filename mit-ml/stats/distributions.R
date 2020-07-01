library(tidyverse)
papers <- as_tibble(read_csv("~/Downloads/CitesforSara.csv"))   
sele<-select(papers,journal, year, cites, title, au1)
cnt<-count(filter(sele,cites>=100))
tst<-group_by(sele,journal)
tst<-summarise(tst,citest = sum(cites))
tstb<-group_by(sele,au1)
tstb<-summarise(tstb,journals = sum(cites))
count(tstb)
head(papers)
papers_female<-select(papers,contains("female"))

#selecting exactly 5 hearts out of 52 cards deck
x<-5
m<-13
n<-39
k<-10
distr<-dhyper(x,m,n,k) 