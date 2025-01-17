import logging
logging.basicConfig(level=logging.INFO)

import statsmodels.api as sm
import os

from skbio.stats.composition import ilr, ilr_inv
from sklearn.linear_model import LinearRegression
import jax
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as onp
#from jax.ops import index, index_update
from jax.lax import lgamma
from classo import classo_problem
from jax import numpy as np
from jax.scipy.special import digamma
from jax_minimize_wrapper import minimize
import skbio.stats.composition as cmp


import sys
sys.path.append(os.getcwd())


function_value = []


class LinearInstrumentModel:
    def __init__(self,
                 X_train,
                 Y_train,
                 Z_train,
                 params_init,
                 lambda_dirichlet,
                 max_iter=20):

        self.params_init_1 = params_init  # should ne a dictionary of alpha0 and alphaT
        self.params_alpha = None  # parameter for alpha estimation
        self.lambda_alpha = None
        self.X = X_train  # should be compositional in our context
        self.Y = Y_train  # outcome
        self.Z = Z_train  # instrument

        self.lambda_dirichlet = lambda_dirichlet  # should be a vector to be able to test more than one
        self._lambda_dirichlet = None  # int which is set in the optimization loop
        self.max_iter = max_iter

        self.n, self.p = X_train.shape  # with n being dimensi
        self.X_n0 = self.add_term_to_zeros(self.X)  # add zeros to the term to be able to perform any log transformation
        self.Xhat = None

        self.params = None

    # setter function
    def set_lambda_dirichlet(self, lam):
        """set lambda in the optimization loop

        Parameters
        ----------
        lam : float
            lambda parameter

        Returns
        -------


        """
        self._lambda_dirichlet = lam

    # method section
    @staticmethod
    def add_term_to_zeros(mat, axis=0, add_term=0.0001):
        """add a small term to the zeros along a specific axis

        Parameters
        -----------
        mat : np.ndarray
            matrix as input
        axis : int={0, 1}
        add_term : float=0.0001
            small count to add to the zero values

        Returns
        -------
        X_n0 : np.ndarray
            matrix with added zero terms

        """
        # TODO : how to implement that sensibly in JAX
        X_n0 = np.zeros(mat.shape)
        for rr in range(len(mat)):
            if mat[rr, :].min() == 0:
                X_n0 = X_n0.at[rr, :].set((mat[rr, :] + add_term) / np.sum(mat[rr, :] + add_term))
                #X_n0 = index_update(X_n0, index[rr, :],  (mat[rr, :] + add_term) / np.sum(mat[rr, :] + add_term))
            else:
                X_n0 = X_n0.at[rr, :].set(mat[rr, :])
                #X_n0 = index_update(X_n0, index[rr, :], mat[rr, :])
        return X_n0


    def gradient_fun_alpha(self, alpha):
        """ this is the gradient of the objective function for alpha

        Parameters
        ----------
        alpha : np.ndarray
            current alpha value during optimization

        Returns
        -------
        grad : np.ndarray
            gradient

        """
        alpha0 = alpha[:self.p]
        alphaT = alpha[self.p:]
        g = np.exp((alpha0 * np.ones((self.n,1))) + (alphaT * self.Z[..., np.newaxis]))
        digamma_vec = np.vectorize(digamma)

        vec = np.tile(digamma_vec(np.sum(g, axis=1)), (self.p, 1)).T - digamma_vec(g) + np.log(self.X_n0)
        S = np.dot((vec * g).T, self.Z)

        alpha_pen = alphaT
        der = self._lambda_dirichlet * np.sign(alpha_pen) * np.abs(alpha_pen)
        grad = np.vstack([-S, -S + der.T])

        return grad.flatten("F")


    def likelihood2(self, params):
        """ log likelihood function for dirichlet regression

        Parameters
        ----------
        params : dict
            dictionary of current alpha parameters for dirichlet regression

        Returns
        -------
        log_likelihood : float
            current log likelihood value during optimization

        """
        alpha0 = params["alpha0"]   # p x 1

        if self.Z.ndim < 2:
            alphaT = params["alphaT"]
            g = np.exp((alpha0 * np.ones((self.n,))).T + (alphaT * self.Z).T)  # n x p
        else:
            helperZ = [params["alphaT" + str(inst)] * self.Z[:, inst] for inst in range(self.Z.shape[1])]
            g = np.exp((alpha0 * np.ones((self.n,))).T + (sum(helperZ)).T)
        lgamma_vec = np.vectorize(lgamma)
        # compute log likelihood
        log_likelihood = np.sum(lgamma_vec(np.sum(g, axis=1)) +
                                np.sum((g - 1) * np.log(self.X_n0) - lgamma_vec(g), axis=1))
        return log_likelihood


    def likelihood2_lassopenalty(self, params):
        """ log likelihood function for the dirichlet regression with lasso penalty for variable selection

        Parameters
        ----------
        params : dict
            dictionary of current alpha parameters for dirichlet regression

        Returns
        -------
        log_likelihood : float
            current penalized log likelihood value during optimization

        """
        if self.Z.ndim < 2:
            alphaT = params["alphaT"]
        else:
            alphaT = np.hstack([params["alphaT" + str(inst)] for inst in range(self.Z.shape[1])])
        log_likelihood_penalty = -self.likelihood2(params) + self._lambda_dirichlet * (np.sum(np.abs(alphaT)))
        return log_likelihood_penalty


    def fit_dirichlet(self, verbose=False):
        """ parameter estimation by a dirichlet regression, estimate alpha coefficient and writes it into the attribute
        params_alpha of the class

        Parameters
        ----------
        verbose : bool=False
            whether to print additional information


        Returns
        -------
        BIC : dict
            BIC value of models for different lambda parameters
        estimates : dict
            log likelihood estimates of models for different lambda parameters
        alpha_params : np.ndarray
            array with best alpha parameters, means for minimial BIC

        """

        BIC = {}
        estimates = {}
        num_inst = 0 if self.Z.ndim < 2 else self.Z.shape[1]

        # set the parameters for the iteration
        params = {"alpha0": self.params_init_1["alpha0"]}
        params.update({"alphaT" + str(inst): self.params_init_1["alphaT" + str(inst)] for inst in range(num_inst)})

        for lam in self.lambda_dirichlet:
            self.set_lambda_dirichlet(lam)
            res_alpha = minimize(self.likelihood2_lassopenalty,
                                 params,
                                 method="SLSQP",
                                 #bounds=bounds,
                                 options={'ftol': 1e-4, 'disp': verbose, "maxiter": self.max_iter})
            # TODO : insert selection of best regularization parameter
            # collect all successful models
            if res_alpha.success:
                params_est = {key: np.round(res_alpha.x[key], 3) for key in res_alpha.x.keys()}
                if self.Z.ndim < 2:
                    alphaT = params_est["alphaT"]
                else:
                    alphaT = np.hstack([params_est["alphaT" + str(inst)] for inst in range(self.Z.shape[1])])
                BIC_alpha = np.log(self.n) * (np.sum(params_est["alpha0"] != 0) + np.sum(alphaT != 0)) - \
                            2 * self.likelihood2(params_est)
                BIC.update({lam: BIC_alpha})
                estimates.update({lam: params_est})
        # choose the best model with the best regularization parameter
        if len(BIC) != 0:
            lam_best = min(BIC, key=BIC.get)
            self.params_alpha = estimates[lam_best]
            self.alpha_lambda = lam_best
            return BIC, estimates, self.params_alpha

    def predict_dirichlet(self, Z=None):
        """make prediction from estimated parameters

        Parameters
        ----------
        Z : np.ndarray=None
            matrix of instrument value entries

        Returns
        -------
        Xhat : np.ndarray
            matrix of estimated compositional X values from first stage dirichlet model

        """
        if Z is None:
            Z = self.Z

        alpha0 = self.params_alpha["alpha0"]  # p x 1
        # p x 1 -> would just need to add more parameters for each new instrument...?

        if Z.ndim < 2:
            alphaT = self.params_alpha["alphaT"]
            g = np.exp((alpha0 * np.ones((self.n,))).T + (alphaT * Z).T)  # n x p
        else:
            helperZ = [self.params_alpha["alphaT" + str(inst)] * Z[:, inst] for inst in range(Z.shape[1])]
            g = np.exp((alpha0 * np.ones((Z.shape[0],))).T + (sum(helperZ)).T)

        g_sum = np.sum(g, axis=1)
        Xhat = np.apply_along_axis(lambda x: x/g_sum, 0, g)
        self.Xhat = Xhat
        return Xhat

    def fit_log_contrast_fast(self, X=None, threshold=0.7):
        """fit log contrast regression with cLasso optimization package

        Parameters
        -----------
        X : np.ndarray=None
            compositional X matrix
        threshold : float
            hyperparameter, threshold for log contrast regression in CLasso

        Returns
        -------
        opt : dict
            optimization results as a struct
        """

        if X is None:
            X = self.Xhat
            X = self.add_term_to_zeros(X)
        X_incl_inter = onp.hstack([onp.array(np.log(X))])
        C = onp.ones((1, len(X_incl_inter[0])))

        opt = classo_problem(X_incl_inter, onp.array(np.squeeze(self.Y)), onp.array(C))
        # specification of minimzation problem
        opt.formulation.intercept = True
        opt.formulation.huber = False
        opt.formulation.concomitant = False
        opt.formulation.classification = False
        opt.model_selection.PATH = True  #
        # opt.model_selection.LAMfixed independent of StabSel
        opt.model_selection.StabSel = True  # Choice, (before it was set to true)
        opt.model_selection.StabSelparameters.method = 'first'
        # opt.model_selection.StabSelparameters.threshold_label = 0.5

        opt.model_selection.StabSelparameters.threshold = threshold

        opt.solve()
        beta = opt.solution.StabSel.refit
        try:
            print(opt.solution)
        except:
            print("Error in computation")

        self.params_beta = {"beta0": np.round(beta[0], 3),
                            "betaT": np.round(beta[1:], 3)}

        return opt

    def predict_log_contrast(self, X=None):
        """predict from the estimated log contrast model

        Parameters
        ----------
        X : np.ndarray=None
            compositional X matrix

        Returns
        -------
        Yhat : np.ndarray
            estimated outcome from second stage estimation

        """
        if X is None:
            X = self.Xhat
        beta0 = self.params_beta["beta0"]
        betaT = self.params_beta["betaT"]
        Yhat = (beta0 * np.ones((X.shape[0],))).T + np.dot(np.log(X), betaT)
        return Yhat




class ALR_Model:
    def __init__(self):
        self.alpha = None
        self.m_0 = None


    def fit(self, X_sim, Z_sim):
        """fit two stage least squares model with alr coordinates, only one stage is fit

        Parameters
        ----------
        X_sim : np.ndarray
            compositional X matrix
        Z_sim : np.ndarray
            matrix with the data entries of the instrumental variable

        Returns
        -------

        """
        p = X_sim.shape[1]
        ZZ_sim = sm.add_constant(Z_sim)
        M = cmp.alr(X_sim)
        res = np.linalg.lstsq(ZZ_sim, M)[0]
        if res.ndim > 1:
            alt_m_0 = res[0, :]
            alt_alpha = res[1:, :]
        else:
            alt_m_0, alt_alpha = res
        # transformation of alt parameters
        self.alpha = cmp.alr_inv(alt_alpha)

        # fit some baseline distribution
        self.m_0 = cmp.alr_inv(alt_m_0)

    def predict(self, Z_sim):
        """predict from estimated parameter

         Parameters
         ----------
         Z_sim : np.ndarray
            matrix with instrumental data entries

         Returns
         -------
         Xhat : np.ndarray
            estimated matrix for first stage alr regression, compositional data

         """
        n = Z_sim.shape[0]
        p = len(self.m_0)

        if Z_sim.squeeze().ndim > 1:
            Xhat = cmp.alr_inv(cmp.alr(self.m_0) * np.ones((n, p - 1)) + Z_sim@cmp.alr(self.alpha))
        else:
            Xhat = cmp.alr_inv(cmp.alr(self.m_0) * np.ones((n, p - 1)) + cmp.alr(self.alpha) * Z_sim)
        return Xhat


def dirichlet_logcontrast(Z, X, Y, Xstar, mle, lambda_dirichlet, max_iter, logcontrast_threshold):
    """Model with 1st stage Dirichlet Regression and 2nd stage LogContrast Regression

    Parameters
    ----------
    Z : np.ndarray
        matrix with instrument entries
    X : np.ndarray
        matrix with compositional data entries
    Y : np.ndarray
        matrix with outcome
    Xstar : np.ndarray
        matrix with interventional compositional data entires
    mle : np.ndarray
        initial alpha_0 value for dirichlet regression
    lambda_dirichlet : np.ndarray
        array of possible lambda values for dirichlet regression, hyperparameter
    max_iter : int
        number of maximum iterations for log contrast regression
    logcontrast_threshold : float
        hyperparameter for log contrast regression between 0 and 1

    Returns
    -------
    beta : np.ndarray
        estimated true beta values
    Xhat : np.ndarray
        estimated Xhat values from the first stage prediction
    Yhat : np.ndarray
        estimated Yhat values from both regressions, causal outcome value

    """
    key = jax.random.PRNGKey(1991)
    n, p = X.shape
    n, num_inst = Z.shape


    params_init = {
       "alpha0": mle[..., np.newaxis],  # np.abs(jax.random.normal(key, (p,1))),
    }
    params_init.update({"alphaT" + str(inst): np.abs(jax.random.normal(key, (p, 1))) for inst in range(num_inst)})
    LinIVModel_sim = LinearInstrumentModel(X, Y, Z, params_init,
                                                  lambda_dirichlet=lambda_dirichlet,
                                                  max_iter=max_iter)

    # 1st Stage
    LinIVModel_sim.fit_dirichlet()
    Xhat = LinIVModel_sim.predict_dirichlet()

    # 2nd Stage
    opt = LinIVModel_sim.fit_log_contrast_fast(threshold=logcontrast_threshold)
    Yhat = LinIVModel_sim.predict_log_contrast(Xstar)

    beta_est_0, beta_est_T = LinIVModel_sim.params_beta.values()
    beta = np.hstack([beta_est_0, beta_est_T])

    print("Beta DirichLetLogContrast: " + str(beta_est_0) + str(beta_est_T))

    return beta, Xhat, Yhat


def r_dirichlet_logcontrast(Z, X, Y, Xstar, logcontrast_threshold):
    """Model with 1st stage Dirichlet Regression and 2nd stage LogContrast Regression

    Parameters
    ----------
    Z : np.ndarray
        matrix with instrument entries
    X : np.ndarray
        matrix with compositional data entries
    Y : np.ndarray
        matrix with outcome
    Xstar : np.ndarray
        matrix with interventional compositional data entires
    mle : np.ndarray
        initial alpha_0 value for dirichlet regression
    lambda_dirichlet : np.ndarray
        array of possible lambda values for dirichlet regression, hyperparameter
    max_iter : int
        number of maximum iterations for log contrast regression
    logcontrast_threshold : float
        hyperparameter for log contrast regression between 0 and 1

    Returns
    -------
    beta : np.ndarray
        estimated true beta values
    Xhat : np.ndarray
        estimated Xhat values from the first stage prediction
    Yhat : np.ndarray
        estimated Yhat values from both regressions, causal outcome value

    """

    # 1st Stage
    Xhat = r_fit_predict_dirichlet(Z, X)

    LinIVModel_sim = LinearInstrumentModel(Xhat, Y, None, None,
                                           lambda_dirichlet=None,
                                           max_iter=None)
    opt_ilr = LinIVModel_sim.fit_log_contrast_fast(Xhat, threshold=logcontrast_threshold)
    Yhat = LinIVModel_sim.predict_log_contrast(Xstar)

    beta_est_0, beta_est_T = LinIVModel_sim.params_beta.values()
    beta = np.hstack([beta_est_0, beta_est_T])

    print("Beta DirichLetLogContrast: " + str(beta_est_0) + str(beta_est_T))

    return beta, Xhat, Yhat


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


def ilr_noregression(Z, X):
    """Model for first stage ilr regression, has no second stage

    Parameters
    ----------
    Z : np.ndarray
        matrix with instrument entries
    X : np.ndarray
        matrix with compositional data entries

    Returns
    -------
    alpha : np.ndarray
        vector for estimated first stage
    Xhat : np.ndarray
        matrix with first stage estimation, compositional data

    """
    # use 2SLS on ilr transformation
    n, p = X.shape
    Z_2sls = sm.add_constant(onp.array(Z))
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = ilr(onp.array(X_n0))
    first = sm.OLS(X_ilr, Z_2sls).fit()
    Xhat = first.predict(Z_2sls)
    alpha = first.params
    return alpha, cmp.ilr_inv(Xhat)


def noregression_ilr(X, Y, Xstar, verbose):
    """Model for second stage ilr regression, has no first stage

    Parameters
    ----------
    X : np.ndarray
        matrix with compositional data entries
    Y : np.ndarray
        matrix with outcome
    Xstar : np.ndarray
        matrix with interventional compositional data entires
    verbose : bool
        logical whether to print information of second stage regression

    Returns
    -------
    beta : np.ndarray
            estimated true beta values
    Yhat : np.ndarray
        estimated Yhat values from both regressions, causal outcome value
    """
    # use 2SLS on ilr transformation
    n, p = X.shape
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = sm.add_constant(ilr(onp.array(X_n0)))
    X_star_ilr = sm.add_constant(ilr(onp.array(Xstar)))
    second = sm.OLS(onp.array(Y), X_ilr).fit()
    if verbose:
        print(second.summary())
    Yhat = second.predict(X_star_ilr)
    beta = second.params
    return beta, Yhat


def ilr_ilr(Z, X, Y, Xstar):
    """Model with 2SLS approach but with a ilr transformed variable, has BOTH stages

    Parameters
    ----------
    Z : np.ndarray
        matrix with instrument entries
    X : np.ndarray
        matrix with compositional data entries
    Y : np.ndarray
        matrix with outcome
    Xstar : np.ndarray
        matrix with interventional compositional data entires

    Returns
    -------
    beta : np.ndarray
            estimated true beta values
    Yhat : np.ndarray
        estimated Yhat values from both regressions, causal outcome value

    """
    # use 2SLS on ilr transformation
    n, p = X.shape
    Z_2sls = sm.add_constant(onp.array(Z))
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = sm.add_constant(ilr(onp.array(X_n0)))
    Y_2sls = onp.array(Y)

    iv2sls_log = IV2SLS(Y_2sls, X_ilr, instrument=Z_2sls).fit()
    print(iv2sls_log.summary())
    Yhat = iv2sls_log.predict(sm.add_constant(ilr(onp.array(Xstar))))
    V = cmp._gram_schmidt_basis(p)
    beta = np.hstack([iv2sls_log.params[0], V.T @ iv2sls_log.params[1:]])

    return beta, Yhat


def ilr_logcontrast(Z, X, Y, Xstar, logcontrast_threshold):
    """Model with 1st stage as OLS regression on ilr transformed components and 2nd stage as log contrast model

    Parameters
    ----------
    Z : np.ndarray
        matrix with instrument entries
    X : np.ndarray
        matrix with compositional data entries
    Y : np.ndarray
        matrix with outcome
    Xstar : np.ndarray
        matrix with interventional compositional data entires
    logcontrast_threshold : float
        hyperparameter for log contrast regression between 0 and 1

    Returns
    -------
    beta : np.ndarray
        estimated true beta values
    Xhat : np.ndarray
        estimated Xhat values from the first stage prediction
    Yhat : np.ndarray
        estimated Yhat values from both regressions, causal outcome value
    """

    Z_2sls = sm.add_constant(onp.array(Z))
    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    X_ilr = ilr(onp.array(X_n0))

    # 1st Stage
    reg = LinearRegression().fit(Z_2sls, X_ilr)
    X_fitted_ilr = reg.predict(Z_2sls)
    Xhat = ilr_inv(X_fitted_ilr)

    LinIVModel_ilr = LinearInstrumentModel(Xhat, Y, None, None,
                                                   lambda_dirichlet=None,
                                                   max_iter=None)
    opt_ilr = LinIVModel_ilr.fit_log_contrast_fast(Xhat, threshold=logcontrast_threshold)
    Yhat = LinIVModel_ilr.predict_log_contrast(Xstar)

    beta_ilr_0, beta_ilr_T = LinIVModel_ilr.params_beta.values()
    # back transformation of ilr beta regression parameters

    beta = np.hstack([beta_ilr_0, beta_ilr_T])

    return beta, Xhat, Yhat


def noregression_logcontrast(Z, X, Y, Xstar, logcontrast_threshold):
    """
    Benchmark model which ignores the instrumental variable approach and fits a log contrast regression with classo
    from X to Y

    Parameters
    ----------
    Z : np.ndarray
        matrix with instrument entries
    X : np.ndarray
        matrix with compositional data entries
    Y : np.ndarray
        matrix with outcome
    Xstar : np.ndarray
        matrix with interventional compositional data entires
    logcontrast_threshold : float
        hyperparameter for log contrast regression between 0 and 1

    Returns
    -------
    beta : np.ndarray
        estimated true beta values
    Yhat : np.ndarray
        estimated Yhat values from both regressions, causal outcome value
    """

    X_n0 = LinearInstrumentModel.add_term_to_zeros(X)
    LinIVModel_benchmark = LinearInstrumentModel(X, Y, None, None,
                                                         lambda_dirichlet=None,
                                                         max_iter=None)
    opt_benchmark = LinIVModel_benchmark.fit_log_contrast_fast(X_n0, threshold=logcontrast_threshold)
    Yhat = LinIVModel_benchmark.predict_log_contrast(Xstar)
    beta_bench_0, beta_bench_T = LinIVModel_benchmark.params_beta.values()
    beta = np.hstack([beta_bench_0, beta_bench_T])

    return beta, Yhat


def nocomp_regression(Z, X, Y, Xstar):
    """
    Benchmark for ignoring compositional nature of the data

    Parameters
    ----------
    Z : np.ndarray
        matrix with instrument entries
    X : np.ndarray
        matrix with compositional data entries
    Y : np.ndarray
        matrix with outcome
    Xstar : np.ndarray
        matrix with interventional compositional data entires

    Returns
    -------
    beta : np.ndarray
        estimated true beta values
    Yhat : np.ndarray
        estimated Yhat values from both regressions, causal outcome value

    """
    from statsmodels.sandbox.regression.gmm import IV2SLS
    sls = IV2SLS(onp.array(Y), sm.add_constant(onp.array(X)), onp.array(Z)).fit()
    Yhat = sls.predict(sm.add_constant(onp.array(Xstar)))
    beta = sls.params

    return beta, Yhat


























