import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../src"))

import numpy as np
import statsmodels.api as sm

import dirichlet
from classo import classo_problem

from method_fct import LinearInstrumentModel


# -----------------------------------------------------------------------------------------
# general utils
# -----------------------------------------------------------------------------------------
def add_pseudo_count(X, epsilon = 0.5):
    """ add small epsilon count to the zeros"""
    X_no = X.copy()
    X_no[X_no==0] = epsilon
    return X_no

def compute_fstatistic(Z, X):
    """computation of f statistic"""

    n = X.shape[0]

    ZZ = np.array(sm.add_constant(Z))
    alphahat = np.linalg.inv(ZZ.T @ ZZ) @ ZZ.T @ X
    MSM = np.sum(((ZZ @ alphahat - np.mean(X, axis=0)) ** 2), axis=0)
    MSE = np.sum(((ZZ @ alphahat - X) ** 2), axis=0) / n
    F = MSM / MSE

    return np.round(F, 2)

# -----------------------------------------------------------------------------------------
# regression fitting
# -----------------------------------------------------------------------------------------


# ----------------------------- Log Contrast Regression -----------------------------------------
def fit_log_contrast_regression(X, Y,
                                threshold=0.7, verbose=True,
                                label=None,
                                use_huber=False,
                                is_classification=False,
                                use_CV=False,
                                use_StabSel=True,
                                use_path=True,
                                fix_lambda=False):
    """ excerpt of log contrast regression """
    X_incl_inter = np.hstack([np.array(np.log(X))])
    C = np.ones((1, len(X_incl_inter[0])))

    opt = classo_problem(X_incl_inter, np.squeeze(Y), np.array(C), label=label)

    # specification of minimzation problem
    opt.formulation.intercept = True
    opt.formulation.huber = use_huber
    opt.formulation.concomitant = False
    opt.formulation.classification = is_classification
    opt.model_selection.PATH = use_path  #

    opt.model_selection.CV = use_CV

    opt.model_selection.StabSel = use_StabSel  # Choice, (before it was set to true)
    opt.model_selection.StabSelparameters.method = 'first'  # change to first / "lam"
    opt.model_selection.StabSelparameters.threshold = threshold

    if fix_lambda:
        opt.model_selection.LAMfixed = True
        opt.model_selection.LAMfixedparameters.rescaled_lam = False
        opt.model_selection.LAMfixedparameters.lam = .001

    opt.solve()

    if verbose:
        print(opt.solution)

    if use_StabSel:
        beta = opt.solution.StabSel.refit
    else:
        beta = opt.solution.CV.refit

    return beta


def predict_log_contrast_regression(X, beta):
    """ prediction of log contrast regression """
    Yhat = (beta[0] * np.ones((X.shape[0],))).T + np.dot(np.log(X), beta[1:])
    return Yhat


def fit_predict_logcontrast(X, Y, threshold=0.7, verbose=True, label=None,
                            use_huber=False,
                            is_classification=False,
                            use_CV=False,
                            use_StabSel=True,
                            use_path=True,
                            fix_lambda=False
                            ):
    beta = fit_log_contrast_regression(X, Y, threshold, verbose, label=label,
                                use_huber=use_huber,
                                is_classification=is_classification,
                                use_CV=use_CV,
                                       use_StabSel=use_StabSel,
                                       use_path=use_path,
                                       fix_lambda=fix_lambda)
    Yhat = predict_log_contrast_regression(X, beta)

    return Yhat, beta


# ----------------------------- Dirichlet Regression -----------------------------------------
def fit_predict_dirichlet(Z, X, Y, lambda_dirichlet=np.array([0.1, 1, 2, 5, 10]), max_iter=500):
    """ fit Dirichlet distribution """

    n, p = X.shape
    n, num_inst = Z.shape

    # estimate mle
    mle = dirichlet.mle(X[np.squeeze(Z == 0)])

    params_init = {
        "alpha0": mle[..., np.newaxis],  # np.abs(jax.random.normal(key, (p,1))),
    }
    params_init.update({"alphaT" + str(inst): np.abs(np.random.normal(size=(p, 1))) for inst in range(num_inst)})

    LinIVModel_sim = LinearInstrumentModel(X, Y, Z, params_init,
                                           lambda_dirichlet=lambda_dirichlet,
                                           max_iter=max_iter)

    # 1st Stage
    LinIVModel_sim.fit_dirichlet()
    # print(LinIVModel_sim.params_alpha)
    Xhat = LinIVModel_sim.predict_dirichlet()

    return Xhat, LinIVModel_sim.params_alpha


def r_fit_predict_dirichlet(Z, X):
    """ fit and predict dirichlet with R package"""
    from rpy2.robjects.packages import importr
    from rpy2.robjects import FloatVector, IntVector
    from rpy2.robjects import r
    import rpy2

    importr("DirichletReg")

    r("""
        fit_predict_dirichlet <- function(Z, X){
            Z_diri <- as.data.frame(Z)
            colnames(Z_diri) <- "Znumeric"
            X_diri <- DR_data(X, base=1)
            res <- DirichReg(X_diri ~ 1 + Znumeric, Z_diri)
            X_hat <- predict(res, Z_diri)
            return(X_hat)
        }
    """)

    n, p = X.shape
    Z_r = r.matrix(IntVector(Z.reshape(-1)), nrow=int(n))
    X_r = r.matrix(FloatVector(X.reshape(-1)), nrow=int(n))
    fit_predict_dirichlet = rpy2.robjects.globalenv["fit_predict_dirichlet"]

    X_diri_hat = fit_predict_dirichlet(Z_r, X_r)
    Xhat = np.asarray(X_diri_hat)

    return Xhat

# ----------------------------- Standard Regression -----------------------------------------
def regression(X, Y):
    """ regression, if necessary, already insert ilr transformation """

    XX = sm.add_constant(X)
    reg = sm.OLS(Y, XX).fit()
    Yhat = reg.predict(XX)

    return Yhat, reg.params
