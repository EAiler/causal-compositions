"""
Run file for computation of confidence interval for different methods

----
Simulated Data is not saved, but will be immediately used for method computation purposes

"""

from typing import Dict, Text, Union

from absl import app
from absl import flags
import jax.numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as onp
from skbio.diversity import alpha_diversity
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import sys
sys.path.append(os.getcwd())

import boundiv
import kiv
from helper_fct import *
from plot_fct import plot_diversity_methods
Results = Dict[Text, Union[float, np.ndarray]]


# ---------------------------- INPUT/OUTPUT -----------------------------------
# I always keep those the same!
flags.DEFINE_string("input_path", "/Users/elisabeth.ailer/Projects/P1_Microbiom/Code/input/data",
                    "Directory of the input data.")
flags.DEFINE_string("save_path", "/Users/elisabeth.ailer/Projects/P1_Microbiom/Code/Figures/RealData",
                    "Path to the output directory (for results).")
flags.DEFINE_string("agg_level", "Genus", "Level of aggregation for the data.")
flags.DEFINE_string("filter", "", "Filter for the data.")

# ------------------------------ MISC -----------------------------------------
# I always keep this one!
flags.DEFINE_integer("seed", 0, "The random seed.")
FLAGS = flags.FLAGS

# local functions


def run_diversity_estimation_methods(Z, X, Y, Ytrue=None, methods=["OLS", "2SLS", "KIV", "Bounds"]):
    """take diversity estimates and run all available methods, prints out first stage F-test to give an indication
    of instrument strength

    Parameters
    ----------
    Z : np.ndarray
        matrix of instrument entries
    X : np.ndarray
        matrix of diversity estimates
    Y : np.ndarray
        matrix of weight entries
    Ytrue : np.ndarray
        matrix of causal weight entries, are standardized the same way as Y is
    methods : list=["OLS", "2SLS", "KIV", "Bounds"]
        list of methods that should be run on the data


    Results
    -------
    x : np.ndarray
        standardized matrix of diversity estimates
    y : np.ndarray
        standardized matrix of outcome entries
    Ytrue : np.ndarray
        standardized matrix of outcome entries for causal Ytrue input
    xstar : np.ndarray
        interventional x value for true causal effect
    xstar_bound : np.ndarray
        interventional x value for bound evaluation
    ystar_ols : np.ndarray
        outcome when estimating causal effect via OLS
    ystar_2sls : np.ndarray
        outcome when estimating causal effect via 2SLS
    ystar_kiv : np.ndarray
        outcome when estimating causal effect via KIV
    results : np.ndarray
        outcome of bounding method

    """

    def whiten_data(col):
        mu = col.mean()
        std = col.std()
        return (col - mu) / std, mu, std

    # whiten the data
    Z, _, _ = whiten_data(Z)
    X, _, _ = whiten_data(X)
    Y, mu_y, mu_std = whiten_data(Y)
    if Ytrue is not None:
        Ytrue = (Ytrue - mu_y)/mu_std

    z = onp.array(Z).squeeze()
    y = onp.array(Y).squeeze()
    x = onp.array(X).squeeze()

    zz = sm.add_constant(z)
    xx = sm.add_constant(x)

    xstar = np.linspace(x.min(), x.max(), 30)

    # F Test
    ols1 = sm.OLS(x, zz).fit()
    xhat = ols1.predict(zz)
    print(ols1.summary())


    # OLS
    if "OLS" in methods:
        ols = sm.OLS(y, xx).fit()
        coeff_ols = ols.params
        ystar_ols = coeff_ols[0] + coeff_ols[1] * xstar
    else:
        ystar_ols = None

    # 2SLS
    if "2SLS" in methods:
        iv2sls = IV2SLS(y, xx, zz).fit()
        coeff_2sls = iv2sls.params
        ystar_2sls = coeff_2sls[0] + coeff_2sls[1] * xstar
    else:
        ystar_2sls = None

    # KIV
    if "KIV" in methods:
        xstar, ystar_kiv = kiv.fit_kiv(z, x, y, xstar=xstar)
    else:
        ystar_kiv = None

    # BoundIV
    if "Bounds" in methods:
        xstar_bound = np.linspace(np.quantile(x, 0.1), np.quantile(x, 0.9), 5)
        satis, results = boundiv.fit_bounds(z, x, y, xstar_bound)
    else:
        xstar_bound = None
        results = None

    return x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results



# =============================================================================
# MAIN
# =============================================================================

def main(_):
    # ---------------------------------------------------------------------------
    # Directory setup, save flags, set random seed
    res = dict()
    input_file = os.path.join(FLAGS.input_path, "divData_" + str(FLAGS.filter) + "_Day_21_" + str(FLAGS.agg_level) \
                              + "_230314.csv")
    data = pd.read_csv(input_file)
    
    Y = data["weight"]
    n = Y.shape[0]
    Z = np.array([data["Treatment"][i].find("control") for i in range(n)]) * (-1)

    # Simpson Diversity Alice
    #X = data["Simpson"]     
    #x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results = run_diversity_estimation_methods(Z,
    #                                                                                                         X, Y)
    #fig = plot_diversity_methods(x, y, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results)
    #fig.update_layout(xaxis=dict(title="Simpson Diversity (standardized)"), showlegend=False, width=1000, height=600)
    #fig.update_yaxes(range=[-2.2, 2.2])
    #fig.update_xaxes(range=[-3, 3])
    #fig.write_image(os.path.join(FLAGS.save_path, FLAGS.agg_level + "_" + FLAGS.filter + "_simpson_div.png"))

    # Shannon Diversity Alice
    #X = data["Shannon"]
    #x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results = run_diversity_estimation_methods(Z,
     #                                                                                                             X, Y)
    #fig = plot_diversity_methods(x, y, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results)
    #fig.update_layout(xaxis=dict(title="Shannon Diversity (standardized)"), showlegend=False, width=1000, height=600)
    #fig.update_yaxes(range=[-2, 2])
    #fig.update_xaxes(range=[-3, 3])
    #fig.write_image(os.path.join(FLAGS.save_path, FLAGS.agg_level + "_" + FLAGS.filter + "_shannon_div.png"))

    # Chao Diversity SciKit Bio
    X = data.iloc[:, 5:-1]
    X = alpha_diversity("chao1", X)
    x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results = run_diversity_estimation_methods(Z,
                                                                                                             X, Y)
    res.update({"chao" :{
    "x" : x, 
    "y" : y, 
    "Ytrue" : Ytrue, 
    "xstar" : xstar, 
    "xstar_bound" : xstar_bound, 
    "ystar_ols" :  ystar_ols, 
    "ystar_2sls" : ystar_2sls, 
    "ystar_kiv" : ystar_kiv, 
    "results" : results
    }})

    # Simpson Diversity SciKit Bio
    X = data.iloc[:, 5:-1]
    X = alpha_diversity("simpson", X)
    x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results = run_diversity_estimation_methods(Z,
                                                                                                             X, Y)
    res.update({"simpson" :{
    "x" : x, 
    "y" : y, 
    "Ytrue" : Ytrue, 
    "xstar" : xstar, 
    "xstar_bound" : xstar_bound, 
    "ystar_ols" :  ystar_ols, 
    "ystar_2sls" : ystar_2sls, 
    "ystar_kiv" : ystar_kiv, 
    "results" : results
    }})

    # Shannon Diversity SciKit Bio
    X = data.iloc[:, 5:-1]
    X = alpha_diversity("shannon", X)
    x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results = run_diversity_estimation_methods(Z,
                                                                                                             X, Y)
    res.update({"shannon" :{
    "x" : x, 
    "y" : y, 
    "Ytrue" : Ytrue, 
    "xstar" : xstar, 
    "xstar_bound" : xstar_bound, 
    "ystar_ols" :  ystar_ols, 
    "ystar_2sls" : ystar_2sls, 
    "ystar_kiv" : ystar_kiv, 
    "results" : results
    }})

    # Fisher Diversity SciKit Bio
    X = data.iloc[:, 5:-1]
    X = alpha_diversity("fisher_alpha", X)
    x, y, Ytrue, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results = run_diversity_estimation_methods(Z,
                                                                                                             X, Y)
    res.update({"fisher" :{
    "x" : x,
    "y" : y,
    "Ytrue" : Ytrue,
    "xstar" : xstar,
    "xstar_bound" : xstar_bound,
    "ystar_ols" :  ystar_ols,
    "ystar_2sls" : ystar_2sls,
    "ystar_kiv" : ystar_kiv,
    "results" : results
    }})

    # save results
    np.save(os.path.join(FLAGS.save_path, FLAGS.agg_level + "_" + FLAGS.filter + "_diversity_estimation.npy"), res)

    
if __name__ == "__main__":
    app.run(main)







































