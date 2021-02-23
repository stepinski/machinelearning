import math
import scipy.stats as st
pi_mle=102/62000
pi_mle_treat=39/31000
pi_mle_control=63/31000
lmbd=-2*math.log(st.binom.pmf(39,31000,102/62000) * st.binom.pmf(63,31000,102/62000) / (st.binom.pmf(39,31000,39/31000) * st.binom.pmf(63,31000,63/31000)))
print(lmbd)