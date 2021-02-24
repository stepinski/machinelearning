import math
import scipy.stats as st
from scipy.stats.distributions import chi2

#likelihood ratio calculation
def likelihood_ratio(t,c,tcount,all):
    return -2*math.log(st.binom.pmf(t,tcount,(t+c)/all) * st.binom.pmf(c,all-tcount,(t+c)/all) / (st.binom.pmf(t,tcount,t/tcount) * st.binom.pmf(c,all-tcount,c/(all-tcount))))

#calc manually
#pi_mle=102/62000
#pi_mle_treat=39/31000
#pi_mle_control=63/31000
#LR=-2*math.log(st.binom.pmf(39,31000,102/62000) * st.binom.pmf(63,31000,102/62000) / (st.binom.pmf(39,31000,39/31000) * st.binom.pmf(63,31000,63/31000)))
LR=likelihood_ratio(39,63,31000,62000)
print(LR)
pvalue = chi2.sf(LR, 1)
print('pvalue %.6f'%pvalue)