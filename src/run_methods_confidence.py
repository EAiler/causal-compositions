"""
Run file for computation of confidence interval for different methods

----
Simulated Data is not saved, but will be immediately used for method computation purposes

"""

from typing import Dict, Text, Union

import json
from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import jax.numpy as np
import pandas as pd
import pickle
import jax
#import logging
#logging.basicConfig(level=logging.INFO)
import os
import numpy as onp
import dirichlet
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import skbio.stats.composition as cmp
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "param_input"))


import boundiv
import kiv
from method_fct import noregression_logcontrast, dirichlet_logcontrast, ilr_logcontrast, ALR_Model
from method_fct import ilr_ilr, r_dirichlet_logcontrast, nocomp_regression
from simulate_data_fct import sim_IV_negbinomial, sim_IV_ilr_linear, sim_IV_ilr_nonlinear
from helper_fct import *

Results = Dict[Text, Union[float, np.ndarray]]


# ---------------------------- INPUT/OUTPUT -----------------------------------
# I always keep those the same!
flags.DEFINE_string("data_dir", "../param_input/",
                    "Directory of the input data.")
flags.DEFINE_string("output_dir", "../results/",
                    "Path to the output directory (for results).")
flags.DEFINE_string("output_name", "",
                    "Name for result folder. Use timestamp if empty.")

# ------------------------------ MISC -----------------------------------------
# I always keep this one!
flags.DEFINE_integer("seed", 0, "The random seed.")
FLAGS = flags.FLAGS

flags.DEFINE_string("paramfile", "paramFile_1", "name of python parameter file.")
flags.DEFINE_integer("n", 10000, "sample number.")
flags.DEFINE_integer("p", 10, "number of compositions/microbes.")
flags.DEFINE_integer("num_inst", 10, "number of instruments.")
flags.DEFINE_integer("num_star", 250, "number of interventional dataset.")
flags.DEFINE_bool("use_data_in_folder", False, "Check if data is already available in the folder, if yes, then use the pickle files in the folder.")
flags.DEFINE_string("add_id", "", "additional identifier, if pre-exisiting data is used, but other methods should be tested.")


flags.DEFINE_list("lambda_dirichlet", [0.1, 1, 2, 5, 10], "List of possible lambda parameters for the dirichlet regression.")
flags.DEFINE_float("logcontrast_threshold", 0.7, "Threshold value for log contrast regression.")
flags.DEFINE_float("kernel_alpha", 1, "Penalization parameter for KIV second stage.")
flags.DEFINE_integer("max_iter", 200, "Maximum number of iterations.")
flags.DEFINE_list("selected_methods", ["OnlyLogContrast", "ILR+LogContrast", "R_DIR+LogContrast",
                                       "DIR+LogContrast", "ILR+ILR"],
                  "List of possible methods to evaluate the dataset for.")
flags.DEFINE_integer("num_runs", 50, "Number of runs for confidence interval computation.")
# local functions


def run_methods_selection(Z_sim, X_sim, Y_sim, X_star, Y_star, beta_ilr_true,
                    lambda_dirichlet, max_iter, logcontrast_threshold, mse_large_threshold=40,
                          kernel_alpha=1,
                          method_selection=["OnlyLogContrast", "ILR+LC", "DIR+LC", "ILR+ILR"]):
    """Run just the most important methods, function for testing and etc.

    Parameters
    ----------
    Z_sim : np.ndarray
        sample matrix of instrument
    X_sim : np.ndarray
        sample matrix of microbiome data, compositional data
    Y_sim : np.ndarray
        sample matrix of output
    X_star : np.ndarray
        sample matrix of interventional microbiome data, compositional data
    Y_star : np.ndarray
        sample matrix of true causal effect based on the interventional data X_star
    mle : np.ndarray
        starting point for dirichlet regression in the first stage
    lambda_dirichlet : float
        penalizing lambda used for dirichlet regression in the first stage
    max_iter : int
        maximum numver if iterations for log contrast regression in the second stage
    mse_large_threshold : int=40
        all data that produces a oos mse over the threshold value is saved in a dictionary attached to mse_large for de-
        bugging purposes
    method_selection : list
        list of methods we want the results for


    Returns
    -------
    mse_all : np.ndarray
        out of sample mean squared error for all methods that have been tested
    beta_all : np.ndarray
        beta values for all methods that have been tested
    title_all : np.ndarray
        array of strings which show which item in mse_all, beta_all belongs to which method
    mse_large : None or dict
        dictionary holding the data which produces a oos mse over the mse_large_threshold value
    """

    n, p = X_sim.shape
    V = cmp._gram_schmidt_basis(p)
    mse_all = []
    title_all = []
    beta_all = []

    # estimate starting point for dirichlet regression
    if p < 10:
        try:
            mle = dirichlet.mle(X_sim[(np.abs(Z_sim) <= 0.2).sum(axis=1) == p, :],
                                tol=.001)
        except:
            mle = np.ones((p,)) / p
    else:
        mle = np.ones((p,)) / p


    # prepare data for kernel regression
    X_sim_ilr = cmp.ilr(X_sim)
    X_star_ilr = cmp.ilr(X_star)

    def whiten_data(X):
        mu, std = X.mean(axis=0), X.std(axis=0)
        X_std = (X - mu) / std
        return X_std, mu, std

    XX, mu_x, std_x = whiten_data(X_sim_ilr)
    YY, mu_y, std_y = whiten_data(Y_sim)
    ZZ, mu_z, std_z = whiten_data(Z_sim)
    ZZ, XX, YY = onp.array(ZZ), onp.array(XX), onp.array(YY)

    # NO COMP RELATION
    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< No Compositional Constraint >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if "NoComp" in method_selection:
        title = "NoComp"

        try:
            betahat, Yhat = nocomp_regression(Z_sim, X_sim, Y_sim, X_star)
            mse = np.mean((Yhat - Y_star) ** 2)
            mse_all.append(mse)
            beta_all.append(V @ betahat[1:])
            logging.info(f"True Beta: " + str(beta_ilr_true))
            logging.info(f"Estimated Beta: " + str(np.round(betahat[1:], 2)))
            logging.info(f"Error: " + str(np.round(mse, 2)))

            title_all.append(title)
            logging.info(f"")
        except:
            mse = np.inf
            betahat = np.array([np.nan] * (p-1))
            mse_all.append(mse)
            beta_all.append(betahat)
            logging.info(f"True Beta: " + str(beta_ilr_true))
            logging.info(f"Estimated Beta: " + str())
            logging.info(f"Error: " + str(np.round(mse, 2)))
            logging.info(f"No solution for " + str(title))

    # NO REGRESSION LOG CONTRAST REGRESSION
    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ONLY Second Stage - Log Contrast >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if "OnlyLogContrast" in method_selection:
        title = "ONLY Second LC"

        try:
            betahat, Yhat = noregression_logcontrast(Z_sim, X_sim, Y_sim, X_star,
                                                     logcontrast_threshold)
            mse = np.mean((Yhat - Y_star) ** 2)
            mse_all.append(mse)
            beta_all.append(V @ betahat[1:])
            logging.info(f"True Beta: " + str(beta_ilr_true))
            logging.info(f"Estimated Beta: " + str(np.round(betahat[1:], 2)))
            logging.info(f"Error: " + str(np.round(mse, 2)))

            title_all.append(title)
            logging.info(f"")
        except:
            logging.info(f"No solution for " + str(title))


    # DIRICHLET LOG CONTRAST REGRESSION
    #if "DIR+LogContrast" in method_selection:
    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - Dirichlet + Log Contrast >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if "DIR+LC" in method_selection:
        if p <= 30:
            try:
                betahat, X_diri_log, Yhat = dirichlet_logcontrast(Z_sim, X_sim, Y_sim, X_star, mle, lambda_dirichlet,
                                                                  max_iter,
                                                                  logcontrast_threshold)

                mse = np.mean((Yhat - Y_star) ** 2)
                mse_all.append(mse)
                beta_all.append(V @ betahat[1:])
                logging.info(f"True Beta: " + str(beta_ilr_true))
                logging.info(f"Estimated Beta: " + str(np.round(V @ betahat[1:], 2)))
                logging.info(f"Error: " + str(np.round(mse, 2)))
                title = "DIR+LC"
                title_all.append(title)
                logging.info(f"")
            except:
                logging.info(f"No solution found for Dirichlet Regression")

        else:
            logging.info(f"Dirichlet not tried due to performance reasons")

    # DIRICHLET LOG CONTRAST REGRESSION
    #if "R_DIR+LogContrast" in method_selection:
    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - Dirichlet + Log Contrast >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if "R_DIR+LC" in method_selection:
        if p <= 30:
            try:
                betahat, X_diri_log, Yhat = r_dirichlet_logcontrast(Z_sim, X_sim, Y_sim, X_star,
                                                                  logcontrast_threshold)

                mse = np.mean((Yhat - Y_star) ** 2)
                mse_all.append(mse)
                beta_all.append(V @ betahat[1:])
                logging.info(f"True Beta: " + str(beta_ilr_true))
                logging.info(f"Estimated Beta: " + str(np.round(V @ betahat[1:], 2)))
                logging.info(f"Error: " + str(np.round(mse, 2)))
                title = "R_DIR+LC"
                title_all.append(title)
                logging.info(f"")
            except:
                logging.info(f"No solution found for Dirichlet Regression")
        else:
            logging.info(f"Dirichlet not tried due to performance reasons")

    # ILR LOG CONTRAST REGRESSION
    #if "ILR+LogContrast" in method_selection:
    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - ILR + Log Contrast >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # ILR REGRESSION LOG CONTRAST REGRESSION
    if "ILR+LC" in method_selection:
        betahat, X_ilr_log, Yhat = ilr_logcontrast(Z_sim, X_sim, Y_sim, X_star,
                                                   logcontrast_threshold)
        mse = np.mean((Yhat - Y_star) ** 2)
        mse_all.append(mse)
        beta_all.append(V @ betahat[1:])
        logging.info(f"True Beta: " + str(beta_ilr_true))
        logging.info(f"Estimated Beta: " + str(np.round(betahat[1:], 2)))
        logging.info(f"Error: " + str(np.round(mse, 2)))
        title = "ILR+LC"
        title_all.append(title)
        logging.info(f"")

    # ILR ILR REGRESSION
    #if "ILR+ILR" in method_selection:
    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - ILR ILR Regression Implementation >>>>>>>>>>>>>>>>>>>>>>>")
    # ILR REGRESSION ILR REGRESSION
    if "ILR+ILR" in method_selection:
        betahat, Yhat = ilr_ilr(Z_sim, X_sim, Y_sim, X_star)
        mse = np.mean((Yhat - Y_star) ** 2)
        mse_all.append(mse)
        beta_all.append(V @ betahat[1:])
        logging.info(f"True Beta: " + str(beta_ilr_true))
        logging.info(f"Estimated Beta: " + str(np.round(V @ betahat[1:], 2)))
        logging.info(f"Error: " + str(np.round(np.mean((Yhat - Y_star) ** 2), 2)))
        title = "ILR+ILR"
        title_all.append(title)
        logging.info(f"")

    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ALR MODEL>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if "ALR+LC" in method_selection:
        first_stage = ALR_Model()
        first_stage.fit(X_sim, Z_sim)
        Xhat_alr2 = first_stage.predict(Z_sim)

        betahat, Yhat = noregression_logcontrast(Z_sim, Xhat_alr2, Y_sim, X_star,
                                                 logcontrast_threshold)
        mse = np.mean((Yhat - Y_star) ** 2)

        mse_all.append(mse)
        beta_all.append(V @ betahat[1:])
        logging.info(f"True Beta: " + str(beta_ilr_true))
        logging.info(f"Estimated Beta: " + str(np.round(V @ betahat[1:], 2)))
        logging.info(f"Error: " + str(np.round(mse, 2)))
        title = "ALR+LC"
        title_all.append(title)
        logging.info(f"")

    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - Kernel Regression KIV >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if "KIV" in method_selection:

        _, Yhat = kiv.fit_kiv(ZZ, XX, YY, xstar=((X_star_ilr - mu_x) / std_x))
        Yhat = std_y * Yhat + mu_y
        mse = np.mean((Yhat - Y_star) ** 2)

        mse_all.append(mse)
        beta_all.append(None)
        logging.info(f"Error: " + str(np.round(mse, 2)))
        title = "KIV"
        title_all.append(title)
        logging.info(f"")


    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<< ONLY SECOND STAGE - Kernel Regression KIV >>>>>>>>>>>>>>>>>>>>>>>>>")
    if "OnlyKIV" in method_selection:

        kernel = "linear"
        from sklearn.kernel_ridge import KernelRidge
        logging.info(f"We use {kernel_alpha}")
        reg = KernelRidge(kernel=kernel, alpha=kernel_alpha).fit(XX, YY)
        Yhat = reg.predict((X_star_ilr - mu_x) / std_x)
        Yhat = std_y * Yhat + mu_y
        mse = np.mean((Yhat - Y_star) ** 2)

        mse_all.append(mse)
        beta_all.append(None)
        logging.info(f"Error: " + str(np.round(mse, 2)))
        title = "ONLY Second KIV"
        title_all.append(title)
        logging.info(f"")


    logging.info(f"---------------------------------------------------------------------------------------------")
    logging.info(f"<<<<<<<<<<<<<<<<<<<<<<<<<<< 2SLS - Kernel Regression KIV (manual) >>>>>>>>>>>>>>>>>>>>>>>>>")
    if "M_KIV" in method_selection:
        from sklearn.kernel_ridge import KernelRidge
        kernel = "linear"
        reg = KernelRidge(kernel=kernel).fit(ZZ, XX)
        Xhat_ilr = reg.predict(ZZ)
        reg2 = KernelRidge(kernel=kernel, alpha=kernel_alpha).fit(Xhat_ilr, YY)
        logging.info(f"We use {kernel_alpha}")
        Yhat = reg2.predict((X_star_ilr - mu_x) / std_x)
        Yhat = std_y * Yhat + mu_y
        mse = np.mean((Yhat - Y_star) ** 2)

        mse_all.append(mse)
        beta_all.append(None)
        logging.info(f"Error: " + str(np.round(mse, 2)))
        title = "Kernel+Kernel"
        title_all.append(title)
        logging.info(f"")


    if any(i > mse_large_threshold for i in mse_all[3:]):
        mse_large = {"X_sim": X_sim,
                     "Y_sim": Y_sim,
                     "X_star": X_star,
                     "Y_star": Y_star}
    else:
        mse_large = None

    return mse_all, beta_all, title_all, mse_large

# =============================================================================
# MAIN
# =============================================================================

def main(_):
    # ---------------------------------------------------------------------------
    # Directory setup, save flags, set random seed
    # ---------------------------------------------------------------------------
    FLAGS.alsologtostderr = True

    if FLAGS.output_name == "":
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        dir_name = FLAGS.output_name
    out_dir = os.path.join(os.path.abspath(FLAGS.output_dir), dir_name)


    logging.info(f"Save all output to {out_dir}...")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    FLAGS.log_dir = out_dir
    logging.get_absl_handler().use_absl_log_file(program_name="run")

    logging.info("Save FLAGS (arguments)...")
    with open(os.path.join(out_dir, 'flags.json'), 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

    logging.info(f"Set random seed {FLAGS.seed}...")

    # ---------------------------------------------------------------------------
    # 1. Load parameter data
    # ---------------------------------------------------------------------------
   # paramfile = os.path.join("param_input", FLAGS.paramfile)
    paramfile = FLAGS.paramfile
    num_star = FLAGS.num_star

    param = __import__(paramfile)
    p = param.p
    num_inst = param.num_inst
    n = param.n

    mse_confidence = []
    title_confidence = []
    beta_confidence = []
    mse_large_confidence = {}

    subkey, _ = jax.random.split(param.key, 2)

    for iter in range(FLAGS.num_runs):
        # we are computing confidence intervals -> different seed for every run
        _, subkey = jax.random.split(subkey, 2)


        logging.info(f"***************************************************************************************")
        logging.info(f"***********************************"+str(iter)+"/"+str(FLAGS.num_runs)+"****************************************")
        logging.info(f"***************************************************************************************")
        # specification of data path
        data_path = os.path.join(out_dir, "data_res_" + param.param_setting + "_n" + str(n) + "_p" + str(p) + "_k" +
                                 str(num_inst) + "_"
                                 + str(iter) + ".pickle")

        # check whether we should use available data
        if FLAGS.use_data_in_folder and (os.path.isfile(data_path)):
            with open(data_path, "rb") as handle:
                data_res = pickle.load(handle)
                Z_sim, X_sim, Y_sim, X_star, Y_star, betaT = data_res["Z_sim"], data_res["X_sim"], data_res["Y_sim"], \
                                                             data_res["X_star"], data_res["Y_star"], data_res["betaT"]
                handle.close()

        else:
            # ---------------------------------------------------------------------------
            # 2. Data Simulation
            # ---------------------------------------------------------------------------
            # ++++++++++++++++++++++++++++ Neg Binomial Simulation ++++++++++++++++++++++++++++
            logging.info(f"Start data simulation.")
            if param.param_setting=="negbinom":
                if p == 3:
                    V = cmp._gram_schmidt_basis(p)
                    alpha0 = param.alpha0
                    alphaT = param.alphaT
                    mu_c = param.mu_c

                    beta0 = param.beta0
                    c_X = param.c_X

                    betaT = param.betaT
                    betaT_log = betaT
                    c_Y = param.c_Y
                    ps = param.ps


                else:
                    V = cmp._gram_schmidt_basis(p)
                    alpha0 = np.hstack(
                        [param.alpha0, jax.random.choice(subkey, np.array([1, 2, 2]), (p - 8,))])

                    if num_inst == p:
                        print("We can simulate a one on one relation between microbiomes and number of instruments")
                        alphaT = np.diag(np.ones(p))
                    else:
                        print("Random simulation of relation between microbes and instruments")
                        alphaT = np.diag(np.ones(p))[:num_inst,:]

                    mu_c = np.hstack([param.mu_c, jax.random.uniform(subkey, (p - 4,), minval=0.01, maxval=0.05)])
                    mu_c = mu_c / mu_c.sum()  # has to be a compositional vector

                    beta0 = param.beta0
                    c_X = param.c_X

                    betaT = np.hstack([param.betaT,
                                       np.zeros((p - 8))])  # beta is chosen to sum up to one
                    c_Y = np.hstack([param.c_Y,
                                     np.zeros((p - 12))])
                    ps = np.hstack([np.zeros((int(p / 2),)), 0.8 * np.ones((p - int(p / 2),))])

                logging.info(f"Start data simulation.")
                start = time.time()
                confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_negbinomial(subkey,
                                                                                     n=n,
                                                                                     p=p,
                                                                                     num_inst=num_inst,
                                                                                     mu_c=mu_c,
                                                                                     c_X=c_X,
                                                                                     alpha0=alpha0,
                                                                                     alphaT=alphaT,
                                                                                     c_Y=c_Y,
                                                                                     beta0=beta0,
                                                                                     betaT=betaT,
                                                                                     num_star=num_star,
                                                                                     ps=ps
                                                                                     )
                time_elapsed = np.round((time.time() - start)/60, 2)
                logging.info(f"This took {time_elapsed} minutes to finish.")

                # if p is small, we have to drop nan values as they might quite frequently occur
                if p==3:
                    idx_keep = ~np.isnan(X_sim)[:, 1]
                    X_sim = X_sim[idx_keep, :]
                    Y_sim = Y_sim[idx_keep]
                    Z_sim = Z_sim[idx_keep, :]
                    idx_keep_star = ~np.isnan(X_star)[:, 1]
                    X_star = X_star[idx_keep_star, :]
                    Y_star = Y_star[idx_keep_star]


            # ++++++++++++++++++++++++++++ High Dimensional Linear ++++++++++++++++++++++++++++
            if param.param_setting=="highdim_linear":

                V = cmp._gram_schmidt_basis(p)
                alpha0 = np.hstack([param.alpha0,  np.zeros(p-1-len(param.alpha0), )]) ###

                if num_inst == p:
                    print("We can simulate a one on one relation between microbiomes and number of instruments")
                    alphaT = np.diag(np.ones(p-1))
                else:
                    print("Random simulation of relation between microbes and instruments")
                    alphaT = np.diag(np.ones(p-1))[:,:num_inst]

                beta0 = param.beta0
                c_X = np.hstack([param.c_X, np.zeros(p-1-len(param.c_X), )])


                betaT_log = np.hstack([param.betaT, np.zeros(p-1-len(param.betaT), )])
                betaT_p = np.hstack([betaT_log, -betaT_log.sum()])
                betaT = V@betaT_p

                logging.info(f"Start data simulation.")
                start = time.time()
                confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_linear(subkey,
                                                                                    n=n,
                                                                                    p=p,
                                                                                    num_inst=num_inst,
                                                                                    mu_c=param.mu_c,
                                                                                    c_X=c_X,
                                                                                    alpha0=alpha0,
                                                                                    alphaT=alphaT,
                                                                                    c_Y=param.c_Y,
                                                                                    beta0=beta0,
                                                                                    betaT=betaT,
                                                                                    num_star=num_star)

                time_elapsed = np.round((time.time() - start) / 60, 2)
                logging.info(f"This took {time_elapsed} minutes to finish.")
            # ++++++++++++++++++++++++++++ Linear Simulation ++++++++++++++++++++++++++++
            if param.param_setting == "linear":
                betaT = param.betaT
                confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_linear(subkey,
                                                                                    n=n,
                                                                                    p=p,
                                                                                    num_inst=num_inst,
                                                                                    mu_c=param.mu_c,
                                                                                    c_X=param.c_X,
                                                                                    alpha0=param.alpha0,
                                                                                    alphaT=param.alphaT,
                                                                                    c_Y=param.c_Y,
                                                                                    beta0=param.beta0,
                                                                                    betaT=betaT,
                                                                                    num_star=num_star)
            # ++++++++++++++++++++++++++++ NON Linear Simulation ++++++++++++++++++++++++++++
            if param.param_setting == "nonlinear":
                betaT = param.betaT
                confounder, Z_sim, X_sim, Y_sim, X_star, Y_star = sim_IV_ilr_nonlinear(subkey,
                                                                                    n=n,
                                                                                    p=p,
                                                                                    num_inst=num_inst,
                                                                                    mu_c=param.mu_c,
                                                                                    c_X=param.c_X,
                                                                                    alpha0=param.alpha0,
                                                                                    alphaT=param.alphaT,
                                                                                    c_Y=param.c_Y,
                                                                                    beta0=param.beta0,
                                                                                    betaT=betaT,
                                                                                    num_star=num_star)

            # ---------------------------------------------------------------------------
            # 2b. Save results of data simulation to folder
            # ---------------------------------------------------------------------------
            data_res={
                "param_setting": param.param_setting,
                "file_param_setting": str(FLAGS.paramfile),
                "Z_sim": Z_sim,
                "X_sim": X_sim,
                "Y_sim": Y_sim,
                "X_star": X_star,
                "Y_star": Y_star,
                "betaT": betaT
            }

            with open(data_path, "wb") as handle:
                pickle.dump(data_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()



        # ---------------------------------------------------------------------------
        # 3. Perform Method computations
        # ---------------------------------------------------------------------------
        logging.info(f"Start method computation.")
        mse_all, beta_all, title_all, mse_large = run_methods_selection(Z_sim, X_sim, Y_sim, X_star, Y_star, betaT,
                                                                  FLAGS.lambda_dirichlet,
                                                                  FLAGS.max_iter,
                                                                  FLAGS.logcontrast_threshold,
                                                                  kernel_alpha=FLAGS.kernel_alpha,
                                                                  method_selection=FLAGS.selected_methods)
        # Save methods in dictionary
        logging.info(f"Append Results.")
        beta_confidence.append(beta_all)
        mse_confidence.append(mse_all)
        title_confidence.append(title_all)
        mse_large_confidence.update({iter: mse_large})


        # ---------------------------------------------------------------------------
        # 4. Aggregation of Results
        # ---------------------------------------------------------------------------
        logging.info(f"Aggregate Results.")
        V = cmp._gram_schmidt_basis(p)
        flatten = lambda t: [item for sublist in t for item in sublist]
        title_all = flatten(title_confidence)
        mse_all = flatten(mse_confidence)
        beta_all = flatten(beta_confidence)

        mse_dict = dict([(key, np.array([])) for key in set(title_all)])

        for i in range(len(title_all)):
            if mse_all[i] is not None:
                value = mse_dict[title_all[i]]
                mse_dict.update({title_all[i]: np.append(value, mse_all[i])})
            else:
                value = mse_dict[title_all[i]]
                mse_dict.update({title_all[i]: np.append(value, np.nan)})

        logging.info(mse_dict)
        
        df_mse = pd.DataFrame(mse_dict)
        df_mse.dropna(axis=1, inplace=True)
        cat_order = df_mse.mean().sort_values().index
        df_mse = df_mse.reindex(cat_order, axis=1)
        df_mse = pd.melt(df_mse, var_name="Method", value_name="MSE")

        beta_all_2 = [jax.numpy.asarray(V.T @ i) if i is not None else jax.numpy.repeat(jax.numpy.nan, p) for i in beta_all]
        df_beta = pd.DataFrame(zip(title_all, beta_all_2), columns=["Method", "Beta"])

        res = {
            "paramfile": paramfile,
            "logthreshold": FLAGS.logcontrast_threshold,
            "df_mse": df_mse,
            "df_beta": df_beta,
            "mse_large": mse_large_confidence,
            "betaT": betaT,
            "beta_all": beta_all,
            "mse_all": mse_all,
            "title_all": title_all
        }

        # ---------------------------------------------------------------------------
        # 5. Save results in pickle file
        # ---------------------------------------------------------------------------
        #with open(os.path.join(out_dir, "res_"+param.param_setting+"_n" + str(n) + "_p" + str(p) + "_k" +
        #                                str(num_inst) +"_numruns"+str(FLAGS.num_runs)+ FLAGS.add_id + ".pickle"),
        #                                "wb") as handle:
        #    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(out_dir, "res_numruns" + str(FLAGS.num_runs) + FLAGS.add_id + ".pickle"),
                  "wb") as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


logging.info(f"DONE")


if __name__ == "__main__":
    app.run(main)







































