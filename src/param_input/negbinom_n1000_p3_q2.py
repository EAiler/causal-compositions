"""
Simple storage of parameter File
can be accessed in the cluster and edited by nano
"""
import numpy as np
import jax

param_setting = "negbinom"

n = 1000
p = 3
num_inst = 2

key = jax.random.PRNGKey(191)

c_X = 2  # dispersion parameter

# instrument strength
alpha0 = np.array([7, 9, 8])

# confounder
mu_c = np.array([0.7, 0.1, 0.2])
mu_c = mu_c / mu_c.sum()  # has to be a compositional vector
 # prob of zero inflation


# relationship between X and Y
beta0 = 1
betaT = np.array([-5, +3, +2])  # beta is chosen to sum up to one

# confounder influence to Y
c_Y = np.array([2, -10, -10])  # confounder is a composition as well.
ps = np.array([0, 0, 0])
alphaT = np.diag(np.ones(p))[:num_inst, :]*5