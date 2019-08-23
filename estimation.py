import numpy as np
import pandas as pd
from scipy import stats

import os
import json
import datetime

import utils


def calc_likelihood(self, p, x, y, link):
    logL = 0
    lam = link.failure_rate(p, x)
    logL = -1*lam + y*np.log(lam) - np.log(special.factorial(y))
    return sum(logL)


def metropolis(p0, x, y, link, sig=1, stop=lambda it: it<100000):
    likelihood = pd.Series()
    
    if type(p0) == type(pd.DataFrame()):
        parameters = p0.copy()
    elif type(p0) == type(pd.Series()):
        parameters = pd.DataFrame(columns=p0.index)
        parameters.loc[0] = p0.tolist()

    likelihood.loc[0] = utils.calc_likelihood(parameters.loc[0], x, y, link)
    
    acceptance = pd.Series()
    acceptance.loc[0] = 1
    
    it = 0
    while stop(it):
        it += 1
        if it % 1000 == 0:
            print it

        for p in parameters.keys():
            parameters.loc[it] = stats.beta
            
        dp = pd.Series(stats.multivariate_normal.rvs(mean=np.zeros((len(sig),)), cov=np.diag(sig.tolist())),
                      index=sig.index)
        p = parameters.loc[it-1] + dp
        
        L = utils.calc_likelihood(p, x, y, link)
        alpha = min(np.exp(L-likelihood.loc[it-1]), 1)        
        accept = np.random.choice([True, False], p=[alpha, 1-alpha])
        
        if accept:
            parameters.loc[it] = p
            acceptance.loc[it] = 1
        else:
            parameters.loc[it] = parameters.loc[it-1]
            acceptance.loc[it] = 0
        
        likelihood.loc[it] = utils.calc_likelihood(parameters.loc[it], x, y, link)
        
    return parameters, acceptance


def transition_function(X, y, link, target_acceptance=0.25, tol=0.03, chain_size=1000):
    
    p = pd.io.json.json_normalize(link.init_params(X, y))
    sig = 0.1*pd.Series(p.loc[0].tolist(), p.keys())
    a = [1,]

    target_acceptance = 0.25
    tol = 0.03

    sigmas = pd.DataFrame(columns=list(p.keys())+['acceptance',])
    params = pd.DataFrame(columns=list(p.keys()))
    it = 0

    while np.abs(np.average(a)-target_acceptance) > tol:
        p, a = metropolis(p, X, y, link, sig=sig, stop=lambda it: it<chain_size)
        a = a.loc[200:]
        sigmas.loc[it] = list(sig.values) + [a.mean(),]
        
        if a.mean()==0:
            pass
        else:
            sig *= a.mean()/target_acceptance
            
        params.loc[it] = p.loc[200:].mean()

        it +=1
        
    best_sigma = (sigmas['acceptance']-target_acceptance).abs().idxmin()

    return sigmas.loc[best_sigma], params.loc[best_sigma]

def get_existing_transition_params(scenario, model_name):
    if not os.path.exists(os.path.join('scenario', scenario, 'chains','metadata.txt')):
        return []
    
    with open(os.path.join('scenario', scenario, 'chains','metadata.txt'), 'r') as f:
        meta = f.read()
        
    if len(meta) == 0:
        return []
    
    meta = json.loads(meta)
    if model_name not in meta.keys():
        return []
    
    sigma = pd.Series(meta[model_name]['sigma'], index=meta[model_name]['variables'])
    p0 = pd.Series(meta[model_name]['p0'], index=meta[model_name]['variables'])
    return sigma, p0

def save_chain_params(sigma, p0, scenario, model_name):
    
    if 'chains' not in os.listdir(os.path.join('scenarios',scenario)):
        os.mkdir(os.path.join('scenarios',scenario,'chains'))

    lib = {model_name: {'variables': p0.index.tolist(), 
                        'sigma': sigma.tolist(), 
                        'p0': p0.tolist(),
                        'time': str(datetime.datetime.now())}}
    print lib
    with open(os.path.join('scenarios',scenario,'chains','metadata.txt'), 'a') as f:
        f.write(json.dumps(lib))