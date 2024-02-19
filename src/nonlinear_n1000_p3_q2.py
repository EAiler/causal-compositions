"""
Simple storage of parameter File
can be accessed in the cluster and edited by nano
"""
import numpy as np
import jax

param_setting = "nonlinear"
key = jax.random.PRNGKey(253)
n=1000
p=3
num_inst=2

c_X = np.array([2, 2])  # dispersion parameter
# c_X = np.array([2, -2])

# instrument strength
alpha0 = np.array([1, 1])
alphaT = np.array([[4, 1],
                    [-1, 3]])

# confounder
mu_c = -1
 # prob of zero inflation


# relationship between X and Y
beta0 = 5
betaT = np.array([6, 2])

# confounder influence to Y
c_Y = 4
# c_Y = 2