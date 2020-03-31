import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

class Link(object):
	def __init__(self):
		return None

	def expand_events(self, x, y):
		idx = y > 1
		events = y.loc[~idx].copy()
		exog = x.loc[~idx].copy()
		for ix, n in y[idx].iteritems():
			events = events.append(pd.Series(np.ones((int(n),))), ignore_index=True)
			exog = exog.append(pd.DataFrame(np.ones((int(n),len(x.keys()))), columns=x.keys()).multiply(x.loc[ix], axis=1), ignore_index=True)
		return events, exog

	def estimate_params(self, events, exog, thresholds={}):
		if len(thresholds.keys()) == len(exog.keys()):
			pass
		else:
			for dim in exog.keys():
				if dim in thresholds.keys():
					pass
				elif dim == 'Wind':
					thresholds[dim] = np.random.choice(np.arange(65,75,1))
				elif dim == 'DayPrecip':
					thresholds[dim] = np.random.choice(np.linspace(0.8,1.0, 20))
				elif dim == 'WindStorm':
					thresholds[dim] = np.random.choice(np.linspace(9,12, 20))
				elif np.percentile(exog[dim], 95) == 0:
					thresholds[dim] = np.percentile(exog[dim][exog[dim]>0], 90)
				else:
					thresholds[dim] = np.percentile(exog[dim], 99)

		exog = exog.subtract(pd.Series(thresholds))

		try:
			clf = LogisticRegression(random_state=0, solver='lbfgs',
									multi_class='multinomial', fit_intercept=False).fit(exog, events)
		except ValueError:
			return [], []

		slopes = dict(zip(exog.keys(), clf.coef_.tolist()[0]))

		for key in slopes.keys():
			if slopes[key] == 0:
				slopes[key] = 1e-5
		return thresholds, slopes

		# old
	# def estimate_params(self, events, exog):
	# 	thresholds = {}
	# 	for dim in exog.keys():
	# 		thresholds[dim] = np.percentile(exog[dim].tolist(), 95)
	# 		if thresholds[dim] == 0:
	# 			thresholds[dim] = np.percentile(exog[dim][exog[dim]>0], 95)
	# 	exog = exog.subtract(pd.Series(thresholds))

	# 	clf = LogisticRegression(random_state=0, solver='lbfgs',
	# 							multi_class='multinomial', fit_intercept=False).fit(exog, events)

	# 	slopes = dict(zip(exog.keys(), clf.coef_.tolist()[0]))

	# 	for key in slopes.keys():
	# 		if slopes[key] == 0:
	# 			slopes[key] = 1e-5
	# 	return thresholds, slopes


	def estimate_cov(self, events, exog, p0, n_partitions=100):
		params = pd.DataFrame(pd.io.json.json_normalize(p0))
		for part in range(n_partitions):
			subset = np.random.choice(events.index, size=len(events)/n_partitions)

			if events.loc[subset].sum() == 0:
				continue

			t, s = self.estimate_params(events.loc[subset], exog.loc[subset])
			if len(t) == 0:
				pass
			else:
				for key in s.keys():
					if s[key] < 0:
						s[key] = 0
				p = {'slope': s, 'threshold': t}
				params = params.append(pd.io.json.json_normalize(p), ignore_index=True)
		return params.cov()

	def init_params(self, x, y, fleet_size):
		events, exog = self.expand_events(x, y)
		thresholds, slopes = self.estimate_params(events, exog)
		p0 = {'slope': slopes,
				'threshold': thresholds}
		cov = self.estimate_cov(events,exog, p0)
		return p0, cov
	

	def failure_prob(self, params, x):
		if type(params) == dict:
			params = pd.Series(params)

		if type(params) == type(pd.DataFrame()):
			# putting this in to fix a bug :-/
			params = params.drop_duplicates()
		else:
			params = pd.DataFrame([params.tolist(),], columns=params.index)

		for i, _params in params.iterrows():
			exponent = pd.Series(0, index=x.index)
			for dim in x.keys():
				if 'threshold.%s'%(dim) in params.keys():
					x_in = x[dim] - (_params.loc['threshold.%s'%(dim)])
					exponent += _params['slope.%s'%(dim)] * x_in
				elif '%s.threshold'%(dim) in params.keys():
					x_in = x[dim] - (_params.loc['%s.threshold'%(dim)])
					exponent += _params['%s.slope'%(dim)] * x_in
				elif 'threshold.%s'%(dim) in params.index:
					x_in = x[dim] - (_params.loc['threshold.%s'%(dim)])
					exponent += _params['slope.%s'%(dim)] * x_in
				elif '%s.threshold'%(dim) in params.index:
					x_in = x[dim] - (_params.loc['%s.threshold'%(dim)])
					exponent += _params['%s.slope'%(dim)] * x_in
		return 1./(1+np.exp(-1*exponent))

        

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
