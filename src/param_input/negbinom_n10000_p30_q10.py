"""
Simple storage of parameter File
can be accessed in the cluster and edited by nano
"""
import numpy as np
import jax

param_setting = "negbinom"

n = 10000
p = 30
num_inst = 10

key = jax.random.PRNGKey(191)

c_X = 2  # dispersion parameter

# instrument strength
alpha0 = np.array([1, 1, 2, 1, 4, 4, 2, 1, 4, 4, 2, 1])

# confounder
mu_c = np.array([0.2, 0.3, 0.2, 0.1])
mu_c = mu_c / mu_c.sum()  # has to be a compositional vector
 # prob of zero inflation


# relationship between X and Y
beta0 = 1
betaT = np.hstack([np.array([-10, -5, -5, -5]), np.array([10, 5, 5, 5])])  # beta is chosen to sum up to one

# confounder influence to Y
c_Y = np.hstack([np.array([10, 10, 5, 15]), np.array([-5, -5, -5, -5, -5, -5, -5, -5])])  # confounder is a composition as well.
