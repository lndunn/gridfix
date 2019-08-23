import numpy as np
import pandas as pd

class Link(object):
    def __init__(self):
        return None
    
    def init_params(self, x, y, pctile=90):
        slope = []
        threshold = []
        for dim in x.keys():
            slope.append(np.log(y.max()-y.mean())/(x[dim].max()-np.percentile(x[dim], pctile)))
            threshold.append(np.percentile(x[dim], pctile))
            
        return {'constant': np.log(np.average(y)),
                'slope': dict(zip(x.keys(), slope)),
                'threshold': dict(zip(x.keys(), threshold))}
    
    def failure_rate(self, params, x):
        log_y = params['constant']*pd.Series(1, index=x.index)
        for dim in x.keys():
            if '%s.threshold'%(dim) in params.keys():
                idx = x[dim] > params['%s.threshold'%(dim)]
                x_in = (x[dim] - params['%s.threshold'%(dim)]).where(idx, other=0)
                log_y += params['%s.slope'%(dim)] * x_in
            elif '%s.slope'%(dim) in params.keys():
                log_y += params['%s.slope'%(dim)] * x[dim]
        return np.exp(log_y)

        

class Constant(object):
    # deprecated
	def __init__(self):
		return None

	def init_params(self, x, y):
		return {'constant': np.log(np.average(y))}


	def failure_rate(self, params, x):
		return np.exp(params['constant']*np.ones((len(x),)))


class Linear(object):
    # deprecated
	def __init__(self):
		return None

	def init_params(self, x, y):
		slope = []
		for dim in x.keys():
			slope.append(np.log(y.max()-y.mean())/(x[dim].max()-np.percentile(x[dim],90)))

		return {'constant': np.log(np.average(y)),
				'slope': dict(zip(x.keys(), slope))}

	def failure_rate(self, params, x):
		log_y = params['constant']*np.ones((len(x),))
		for dim in x.keys():
			log_y += params['slope.%s'%(dim)] * x[dim]
		return np.exp(log_y)


class Piecewise(object):
    # deprecated
	def __init__(self):
		return None

	def init_params(self, x, y, pctile=90):
		slope = []
		threshold = []
		for dim in x.keys():
			slope.append(np.log(y.max()-y.mean())/(x[dim].max()-np.percentile(x[dim], pctile)))
			threshold.append(np.percentile(x[dim], pctile))
		return {'constant': np.log(np.average(y)),
				'slope': dict(zip(x.keys(), slope)),
				'threshold': dict(zip(x.keys(), threshold))}

	def failure_rate(self, params, x):
		log_y = params['constant']*np.ones((len(x),))
		for dim in x.keys():
			idx = x[dim] < params['threshold.%s'%(dim)]
			x_in = (x[dim] - params['threshold.%s'%(dim)]).where(idx, 0)
			log_y += params['slope.%s'%(dim)] * x_in
		return np.exp(log_y)

def list_all():
    # deprecated
	return ['Constant', 'Linear', 'Piecewise']
