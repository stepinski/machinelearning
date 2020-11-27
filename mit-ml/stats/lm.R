setwd("~/ws/prj/machinelearning/mit-ml/stats/")
test_Data<-read.csv(file="nlsw88.csv")
head(test_Data)
lmodel<-lm(lwage~yrs_school,data=test_Data)
summary(lmodel)$coefficients[,4]  
coefficients(lmodel) # model coefficients
ci <- confint(lmodel, level=0.9) 
ci
z <- (avgs - mu)/sqrt(sample.variance)
qqnorm(z)
qqline(z)
lmodel<-lm(lwage~black,data=test_Data)
summary(lmodel)$coefficients[,4]  
coefficients(lmodel) # model coefficients
ci <- confint(lmodel, level=0.98) 
ci

summary(lmodel)

#multivariable regression
multi <- lm(lwage ~ yrs_school + ttl_exp, data = test_Data)
summary(multi) # show results
anova_unrest <- anova(multi)

#Restricted model
test_Data$newvar <- test_Data$yrs_school + 2*test_Data$ttl_exp
restricted <- lm(lwage ~ newvar, data = test_Data)
summary(restricted) # show results
anova_rest <- anova(restricted)

#Test
statistic_test <- (((anova_rest$`Sum Sq`[2]-anova_unrest$`Sum Sq`[3]/1)/((anova_unrest$`Sum Sq`[3]/anova_unrest$Df[3]))))
statistic_test
pvalue <- df(statistic_test, 1, anova_unrest$Df[3])
pvalue 


z <- (avgs - mu)/sqrt(sample.variance)
qqnorm(z)
qqline(z)