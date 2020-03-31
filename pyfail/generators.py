import numpy as np
import pandas as pd

from scipy import stats
import json


def params(mean, fleet_size, variance=None, distributions=[], names=None, seed=0):
    if type(mean) == list or type(mean) == dict:
        pass
    else:
        mean = [mean,]
        
    params = pd.DataFrame(columns=range(len(mean)), index=range(fleet_size))
    
    for p in params.keys():
        if (len(distributions) == 0) or (distributions[p] == None):
            params[p] = mean[p]
        else:
            sampler = distributions[p](mean[p], variance[p])
            params[p] = sampler.rvs(size=fleet_size, random_state=seed)

    if not names == None:
        params = params.rename(columns=dict(zip(params.keys(), names)))
    return params

# def OLDparams(scenario, fleet_size=10000, seed=0):
    
#     hyperpriors = {}
#     keys = pd.DataFrame([key.split('_') for key in scenario.keys()]).set_index(0)
    
#     params = pd.DataFrame(columns=list(set([key.split('_')[0] for key in scenario.keys()])), 
#                           index=range(fleet_size))

#     for p in params.keys():
#         distribution_name = scenario[p+'_distribution']
#         if pd.isnull(distribution_name):
#             continue
            
#         if distribution_name == 'constant':
#             params[p] = float(scenario[p+'_hyperparams'])
            
#         elif distribution_name == 'norm':
#             sampling_distribution = getattr(stats, distribution_name)

#             hyperparams = eval(scenario[p+'_hyperparams'])
#             sampling_distribution = sampling_distribution(*hyperparams)

#             params[p] = sampling_distribution.rvs(size=fleet_size, random_state=seed)
#             params[p] = params[p].round(4)

#         elif distribution_name == 'GMM':
#             print scenario[p+'_hyperparams']
#             p_spec = json.loads(scenario[p+'_hyperparams'])
#             p_vals = []
#             for model in p_spec:
#                 sampling_distribution = getattr(stats, model['distribution'])
#                 sampling_distribution = sampling_distribution(*model['params'])

#                 n_components = int(model['weight'] * fleet_size)
#                 p_vals.extend(sampling_distribution.rvs(size=n_components, random_state=seed).tolist())
             
#             params[p] = p_vals
#             if pd.isnull(params[p].iloc[-1]):
#                 params[p].iloc[-1] = params[p].iloc[-2]
#                 assert pd.notnull(params[p].iloc[-1])
#     return params


def failure_data(params, X, link, seed=0, fleet_size=1000):
    # remove null parameters
    for p in params.keys():
        if pd.isnull(params[p].loc[0]):
            params = params.drop(p, axis=1)
    
    n_components = params.groupby(params.keys().tolist())[params.keys()[0]].count()
    p_labels = params.keys()
    params = params.set_index(params.keys().tolist())
    
    # note that for a Poisson process Y where Y is the sum of Poisson processes...
    #        Y = Y_1 + Y_2 + Y_3 + Y_4
    # The rate l is also the sum of the rates
    #        l = l_1 + l_2 + l_3 + l_4
    # So the failure rate for the system overall is just the sum of the failure rates
    # for the individual components
    failure_rate = pd.Series(0, index=X.index)
    for p_vals in params.index.unique():
        failure_prob = pd.Series(link().failure_prob(dict(zip(p_labels, p_vals)), X), index=X.index)
        failure_rate += n_components.loc[p_vals] * failure_prob

    failure_realization = pd.Series(index=X.index)

    for rate in failure_rate.drop_duplicates():
        idx = failure_rate == rate
        failure_realization.loc[idx] = stats.poisson.rvs(rate, size=sum(idx), random_state=seed)

    failure_realization.loc[failure_realization>fleet_size] = fleet_size
    return failure_realization, failure_rate