"""
Simple storage of parameter File
can be accessed in the cluster and edited by nano
"""
import numpy as np
import jax

param_setting = "linear"
key = jax.random.PRNGKey(253)
n = 1000
p = 3
num_inst = 2

c_X = np.array([1, 1])  # dispersion parameter

# instrument strength
alpha0 = np.array([4, 1])
alphaT = np.array([[+0.15, 0.15],
                    [0.2, 0]])

# confounder
mu_c = -2
 # prob of zero inflation


# relationship between X and Y
beta0 = 2
betaT = np.array([6, 2])

# confounder influence to Y
c_Y = 4



#c_X = np.array([0.5, 0.5])  # dispersion parameter
#
## instrument strength
#alpha0 = np.array([1, 1])
#alphaT = np.array([[+0, 0.1],
#                    [0.2, 0]])
#
## confounder
#mu_c = -3
# # prob of zero inflation
#
#
## relationship between X and Y
#beta0 = 5
#betaT = np.array([4, 1])
#
## confounder influence to Y
#c_Y = 4
