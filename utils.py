import numpy as np
from scipy import special

def calc_likelihood(p, x, y, link):
    logL = 0
    lam = link.failure_rate(p, x)
    logL = -1*lam + y*np.log(lam) - np.log(special.factorial(y))
    return sum(logL)



def generate_realization(self, x, failure_process, seed=0):
    # DEPRECATED - generators.py now handles all things stochastic
    y = pd.Series(index=x.index)
    rate = failure_process.link_function(failure_process.params, x)
    for r in np.unique(rate):
        idx = rate==r
        y.loc[idx] = stats.poisson.rvs(r, size=sum(idx), random_state=seed)
    return y

