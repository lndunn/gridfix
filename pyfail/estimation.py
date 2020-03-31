import numpy as np
import pandas as pd
from scipy import stats

import os
import json
import datetime

import utils


def calc_likelihood(params, x, y, link, fleet_size):
    logL = 0
    p = link.failure_prob(params, x)
    lam = fleet_size * p
    logL = -1*lam + y*np.log(lam) - np.log(special.factorial(y))
    return sum(logL)


def metropolis(p0, x, y, link, fleet_size, cov=None, jump=None, accept_reject=True, stop=lambda it: it<1000000):
    likelihood = pd.Series()
    if type(p0) == type(pd.DataFrame()):
        parameters = p0.copy()
    elif type(p0) == type(pd.Series()):
        parameters = pd.DataFrame(columns=p0.index)
        parameters.loc[0] = p0.tolist()

    idx = parameters < 0
    parameters[idx] = 0.01

    likelihood.loc[0] = utils.calc_likelihood(parameters.loc[0], x, y, link, fleet_size)

    acceptance = pd.Series()
    acceptance.loc[0] = 1
    
    it = 0
    while stop(it):
        it += 1
        if it % 1000 == 0:
            print (it)

        if type(jump) == type(None):
            dp = pd.Series(stats.multivariate_normal.rvs(mean=np.zeros((len(cov),)), cov=cov),
                          index=parameters.keys())
            p = parameters.loc[it-1] + dp
        else:
            p = jump(parameters.loc[it-1], cov)

        L = utils.calc_likelihood(p, x, y, link, fleet_size)
        
        if accept_reject:
            alpha = min(np.exp(L-likelihood.loc[it-1]), 1)
            accept = np.random.choice([True, False], p=[alpha, 1-alpha])
        else:
            accept = True

        if accept:
            parameters.loc[it] = p
            acceptance.loc[it] = 1
        else:
            parameters.loc[it] = parameters.loc[it-1]
            acceptance.loc[it] = 0
        
        likelihood.loc[it] = utils.calc_likelihood(parameters.loc[it], x, y, link, fleet_size)
        
    return parameters, acceptance


def transition_function(X, y, link, fleet_size, target_acceptance=0.25, tol=0.03, chain_size=1000):
    p0, cov = link.init_params(X, y, fleet_size)
    p = pd.io.json.json_normalize(p0)
    a = [1,]

    covs = []
    acceptance = pd.Series()
    params = pd.DataFrame(columns=list(p.keys()))
    it = 0

    while np.abs(np.average(a)-target_acceptance) > tol:
        p, a = metropolis(p, X, y, link, fleet_size, cov=cov, stop=lambda it: it<chain_size)
        a = a.loc[200:]
        
        covs.append(cov)
        acceptance.loc[it] = a.mean()
        params.loc[it] = p.loc[200:].mean()        
        
        if a.mean()==0:
            pass
        else:
            cov *= a.mean()/target_acceptance

        it +=1
        
    best_iteration = (acceptance-target_acceptance).abs().idxmin()

    return covs[best_iteration], params.loc[best_iteration]

def get_existing_transition_params(scenario, model_name):
    if not os.path.exists(os.path.join('scenarios', scenario, 'chains','metadata.txt')):
        return []
    
    with open(os.path.join('scenarios', scenario, 'chains','metadata.txt'), 'r') as f:
        meta = f.read()
        
    if len(meta) == 0:
        return []
    
    try:
        meta = json.loads('[%s]'%(meta.replace('}{', '},{')))
    except:
        return []

    for m in meta:
        if model_name in m.keys():
            m = m[model_name]
            sigma = pd.DataFrame(np.array(m['sigma']), columns=m['variables'], index=m['variables'])
            p0 = pd.Series(m['p0'], index=m['variables'])
            return sigma, p0
    
    return []

    
def save_chain_params(sigma, p0, scenario, model_name):
    
    if 'chains' not in os.listdir(os.path.join('scenarios',scenario)):
        os.mkdir(os.path.join('scenarios',scenario,'chains'))

    lib = {model_name: {'variables': p0.index.tolist(), 
                        'sigma': np.array(sigma).tolist(), 
                        'p0': p0.tolist(),
                        'time': str(datetime.datetime.now())}}

    with open(os.path.join('scenarios',scenario,'chains','metadata.txt'), 'a') as f:
        f.write(json.dumps(lib))




def hybrid_gibbs_sampler(theta, cov, design_lims=[40,50]):
    # this function didn't work :-(
    theta_star = pd.Series(index=theta.index)
    theta_star.loc['threshold.Wind'] = stats.uniform.rvs(*design_lims)
    for param in ['threshold.Precip','threshold.WindStorm']:
        if param in theta_star.index:
            key = param.split('.')[1]
            idx = X[key] > 0
            theta_star.loc[param] = stats.uniform.rvs(X[key][idx].min(), X[key][idx].max())

    for _var in var:
        A = 'slope.%s'%(_var)
        B = 'threshold.%s'%(_var)
        cov_inv = pd.DataFrame(np.linalg.inv(cov), columns=cov.keys(), index=cov.index)

        conditional_mean = theta.loc[A] + cov[A].loc[B]*cov_inv[B].loc[B]*(theta_star.loc[B] - theta.loc[B])
        conditional_var = cov[A].loc[A] - cov[A].loc[B]*cov_inv[B].loc[B]*cov[B].loc[A]

        theta_star.loc[A] = stats.norm.rvs(conditional_mean, np.sqrt(conditional_var))
    return theta_star