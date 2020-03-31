import numpy as np
import pymc

# specify priors
alpha = pymc.Uniform('alpha', 0,100, value=np.average(50))
beta = pymc.Uniform('beta', 0.2, value=1)

# define link function relating failures to conditions X
@pymc.deterministic
def model(a=alpha, b=beta, x=X['Wind'], s=fleet_size):
    return s/(1+np.exp(-1*b*(x-a)))

failures = pymc.Poisson('failures', mu=model, value=failures, observed=True)