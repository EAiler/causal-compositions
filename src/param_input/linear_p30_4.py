"""
Simple storage of parameter File
can be accessed in the cluster and edited by nano
"""
import numpy as np
import jax

param_setting = "highdim_linear"

n = 10000
p = 30
num_inst = 10

key = jax.random.PRNGKey(191)

c_X = np.array([2, 2, 2, 1, 1, -2, -2, -2, -1, -1])  # dispersion parameter

# instrument strength
alpha0 = np.array([2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1])

# confounder
mu_c = 3

 # prob of zero inflation


# relationship between X and Y
beta0 = 5
betaT = np.array([5, 5, 5, 3, 3, -5, -5, -5, 3, 3])  # beta is chosen to sum up to one

# confounder influence to Y
c_Y = 5  # confounder is a composition as well.