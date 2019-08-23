import numpy as np
from scipy import stats


scenarios = [{'params': {'constant': stats.norm(loc=1, scale=1e-6),
						'slope.Wind': stats.norm(loc=1, scale=1e-6),
						'threshold.Wind': stats.norm(loc=20, scale=1e-6) 
						},
			  'model': 'Constant'},

			 {'params': {'constant': stats.norm(loc=0.1, scale=1e-6),
						'slope.Wind': stats.norm(loc=1, scale=1e-6),
						'threshold.Wind': stats.norm(loc=20, scale=1e-6) 
						},
			  'model': 'Constant'},
			
			 {'params': {'constant': stats.norm(loc=.01, scale=1e-6),
						'slope.Wind': stats.norm(loc=1, scale=1e-6),
						'threshold.Wind': stats.norm(loc=20, scale=1e-6) 
						},
			  'model': 'Constant'},
			
			 {'params': {'constant': stats.norm(loc=1, scale=1e-6),
						'slope.Wind': stats.norm(loc=1, scale=1e-6),
						'threshold.Wind': stats.norm(loc=20, scale=1e-6) 
						},
			  'model': 'Piecewise'},


			  ]