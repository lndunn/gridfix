import numpy as np
import pandas as pd
from scipy import stats

import os
import json
import datetime

import failuremodes.utils


def fix_naming_issue(params, keys):

	for key in list(params.keys()):
		if key.split('.')[0] in keys:
			new_label = '%s.%s'%(key.split('.')[1], key.split('.')[0])
			params = params.rename(columns={key: new_label})

	return params

def calc_grid_area(param_vals, gridsize):
	dp = 1
	for key in list(param_vals.keys()):
		if param_vals[key].min()==param_vals[key].max():
			continue
		else:
			dp *= (param_vals[key].max()-param_vals[key].min())/(float(gridsize)-1)
	return dp

def param_distributions(params, true=False):
	if true:
		return params.mean(), params.cov()
	else:
		tol = 0.1
		bins = np.arange(params.index[0], params.index[-1], 5000)
		target = params.loc[bins[-2]:].mean()
		ix = 0
		mean = params.loc[bins[ix]:bins[ix+1]].mean()
		while ((mean - target) > tol).any():
			ix += 1
			if ix + 1 == len(bins):
				print('MARKOV CHAIN DID NOT CONVERGE')
				return np.nan, np.nan
			mean = params.loc[bins[ix]:bins[ix+1]].mean()
		means = params.loc[bins[ix]:].mean()
		cov = params.loc[bins[ix]:].cov()
		return means, cov


def param_grid(mean, cov, gridsize=10):
	grid = {}
	for key in mean.index:
		if cov[key].loc[key] < 1e-2:
			grid[key] = np.linspace(mean.loc[key], mean.loc[key], 1)
		else:
			p_min = mean.loc[key]-4*cov[key].loc[key]
			p_max = mean.loc[key]+4*cov[key].loc[key]
			grid[key] = np.linspace(p_min, p_max, gridsize)

	pvals = np.meshgrid(*list(grid.values()))
	p_array = pd.DataFrame(list(zip(*[p.flatten() for p in pvals])), columns=list(cov.keys()))
	return p_array


def param_probs(mean, cov, grid):
	for key in list(grid.keys()):
		if grid[key].min()==grid[key].max():
			grid = grid.drop(key, axis=1)
			mean = mean.drop(key, axis=0)
			cov = cov.drop(key, axis=1)
			cov = cov.drop(key, axis=0)

	if grid.shape[1]==0:
		return np.array([[1],])
	pdf = stats.multivariate_normal.pdf(np.array(grid), mean=np.array(mean), cov=np.array(cov))
	return pdf

def likelihood(param_grid, param_prob, x, y, param_names, fleet_size=10000):
	logL = pd.Series()
	for i, p in param_grid.iterrows():
		logL.loc[i] = utils.calc_likelihood(p, X, y['count'], links.Link(), 10000)

	return
