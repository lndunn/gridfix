import numpy as np
from scipy import special
import pandas as pd

from matplotlib import pyplot as plt

def calc_likelihood(params, x, y, link, fleet_size):
    logL = 0
    p = link.failure_prob(params, x)
    lam = fleet_size * p
    assert not any(pd.isnull(lam))
    assert not any(lam==0)
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

