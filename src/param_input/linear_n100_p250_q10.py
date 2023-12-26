"""
Simple storage of parameter File
can be accessed in the cluster and edited by nano
"""
import numpy as np
import jax

param_setting = "highdim_linear"

n = 100
p = 250
num_inst = 10

key = jax.random.PRNGKey(191)

c_X = np.array([-1, 2, -1, 2, -1, 2, -2, 1, -2, 1, -2, 1])  # dispersion parameter

# instrument strength
alpha0 = np.array([1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1])

# confounder
mu_c = 3

# relationship between X and Y
beta0 = 5
betaT = np.array([10, 5, 5, 5, -10, -5, -5, -5])  # beta is chosen to sum up to one

# confounder influence to Y
c_Y = 5  # confounder is a composition as well.
