"""Stripped down re-implementation kernel instrumental variable (KIV)."""

from absl import logging

import numpy as np
from scipy.optimize import minimize

from sklearn.model_selection import train_test_split

from typing import Tuple, Text, Dict

Kerneldict = Dict[Text, np.ndarray]


def median_inter(x: np.ndarray) -> float:
    A = np.repeat(x[:, np.newaxis], len(x), -1)
    dist = np.abs(A - A.T).ravel()

    return (float(np.median(dist)))

def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error.
    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray

    Returns
    -------
    mse : float

    """
    return float(np.mean((x - y) ** 2))


def make_psd(k: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Make matrix positive semi-definite.
    Parameters
    ----------
    k : np.ndarray
        matrix to make positive semi-definite
    eps : float=1e-10
        precision parameter

    """
    n = k.shape[0]
    return (k + k.T) / 2 + eps * np.eye(n)


def get_k(x: np.ndarray,
          y: np.ndarray,
          z: np.ndarray,
          x_vis: np.ndarray) -> Kerneldict:
    """Setup all required matrices from input data.

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    z : np.ndarray
    x_vis : np.ndarray


    Returns
    -------
    results : dict
        dictionary with different kernels and combinations

    """
    vx = [median_inter(x_col) for x_col in x.T] if x.ndim > 1 else [median_inter(x)]
    vz = [median_inter(z_col) for z_col in z.T] if z.ndim > 1 else [median_inter(z)]
    x1, x2, y1, y2, z1, z2 = train_test_split(
        x, y, z, shuffle=False, test_size=0.5)

    results = {
        'y1': y1,
        'y2': y2,
        'y': y,
        'K_XX': get_k_multi_matrix(x1, x1, vx),
        'K_xx': get_k_multi_matrix(x2, x2, vx),
        'K_xX': get_k_multi_matrix(x2, x1, vx),
        'K_Xtest': get_k_multi_matrix(x1, x_vis, vx),
        'K_ZZ': get_k_multi_matrix(z1, z1, vz),
        'K_Zz': get_k_multi_matrix(z1, z2, vz),
    }
    return results


def get_k_multi_matrix(x1: np.ndarray, x2: np.ndarray, v: list, agg_method: str = "additive") -> np.ndarray:
    """ return multivariate kernel product matrix

    Parameters
    ----------
    x1 : np.ndarray
    x2 : np.ndarray
    v : list
    agg_method : str={"additive", "product"}
        additive or multiplicative kernel aggregation methods

    Returns
    -------
    K_multi : cumulative kernel
    """
    K_multi = np.ones((x1.shape[0], x2.shape[0]))
    x1 = x1[:, np.newaxis] if x1.ndim < 2 else x1  # enlarge array for being able to proceed computations
    x2 = x2[:, np.newaxis] if x2.ndim < 2 else x2  # enlarge array for being able to proceed computations

    for i in range(x1.shape[1]):
        for j in range(x2.shape[1]):
            if agg_method == "product":
                K_multi *= get_k_matrix(x1[:, i], x2[:, j], v[i])
            if agg_method == "additive":
                K_multi += get_k_matrix(x1[:, i], x2[:, j], v[i])
            else:
                # TODO : make this valid for callable
                logging.warning("Specify multidimensional kernel function")
    return K_multi


def get_k_entry(x1: np.ndarray, x2: np.ndarray, v: float) -> np.ndarray:
    """ get k entry, calculates entry of the kernel matrix, uses radial basis function

    Parameters
    ----------
    x1 : np.ndarray
    x2 : np.ndarray
    v : float

    Returns
    -------
    k_entry : float
        entry of kernel matrix
    """
    return np.exp(- (np.linalg.norm(x1 - x2, 2) ** 2) / (2. * v ** 2))


def get_k_matrix(x1: np.ndarray, x2: np.ndarray, v: float) -> np.ndarray:
    """Construct rbf kernel matrix with parameter v.

    Parameters
    ----------
    x1 : np.ndarray
    x2 : np.ndarray
    v : float


    Returns
    -------
    k_entry : float

    """
    m = len(x1)
    n = len(x2)

    if len(set(np.array(x1))) < 3:
        v = 0.0001 #.001  
    v = 15 #.001
    #v = 0.001
    
    #print("v", v)

    x1 = np.repeat(x1[:, np.newaxis], n, 1)
    x2 = np.repeat(x2[:, np.newaxis].T, m, 0)
    return np.exp(- ((x1 - x2) ** 2) / (2. * v ** 2))


def kiv1_loss(df: Kerneldict, lam: float) -> float:
    """Loss for tuning hyperparameter lambda.

    Parameters
    -----------
    df : Kerneldict
        dictionary of relevant kernels
    lam : float
        hyperparameter lambda

    Returns
    -------


    """
    n = len(df["y1"])
    m = len(df["y2"])

    brac = make_psd(df["K_ZZ"]) + lam * np.eye(n)
    gamma = np.linalg.solve(brac, df["K_Zz"])
    return np.trace(df["K_xx"] - 2. * df["K_xX"] @ gamma +
                    gamma.T @ df["K_XX"] @ gamma) / m


def kiv2_loss(df: Kerneldict, lam: float, xi: float) -> float:
    """Loss for tuning hyperparameter xi."""
    y1_pred = kiv_pred(df, lam, xi, 2)
    return mse(df["y1"], y1_pred)


def kiv_pred(df: Kerneldict, lam: float, xi: float, stage: int) -> np.ndarray:
    """Kernel instrumental variable prediction."""
    n = len(df["y1"])
    m = len(df["y2"])

    brac = make_psd(df["K_ZZ"]) + lam * np.eye(n)
    W = np.linalg.solve(brac, df["K_XX"]).T @ df["K_Zz"]
    brac2 = make_psd(W @ W.T) + m * xi * make_psd(df["K_XX"])
    alpha = np.linalg.solve(brac2, W @ df["y2"])

    if stage == 2:
        k_xtest = df["K_XX"]
    elif stage == 3:
        k_xtest = df["K_Xtest"]
    else:
        raise ValueError(f"Stage must be 2 or 3, not {stage}")
    return (alpha.T @ k_xtest).T


def fit_kiv(z: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            lambda_guess: float = None,
            xi_guess: float = None,
            fix_hyper: bool = False,
            xstar : np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """ Fit kernel instrumental variable regression.

  Args:
    z: Instrument
    x: Treatment
    y: Outcome
    num_xstar: Number of points to put on the x grid
    lambda_guess: Guess for lambda. Either starting point for optimization or
        fixed if `fix_hyper` is True.
    xi_guess: Guess for xi. Either starting point for optimization or fixed
        if `fix_hyper` is True.
    fix_hyper: Whether to use fixed hyperparameters instead of optimizing.

  Returns:
    xstar: linear grid over range of provided x values
    ystar: predicted treatment effect evaluated at x_star
  """
    logging.info("Setup matrices...")
    df = get_k(x, y, z, xstar)

    if not fix_hyper:
        lambda_0 = lambda_guess or np.log(0.05)

        def kiv1_obj(lam: float):
            return kiv1_loss(df, np.exp(lam))

        logging.info("Optimize lambda...")
        res = minimize(kiv1_obj, x0=lambda_0, method='BFGS')
        if not res.success:
            logging.info("KIV1 minimization did not succeed.")
        lambda_star = res.x
        logging.info(f"Optimal lambda {lambda_star}...")
    else:
        if lambda_guess is None:
            raise ValueError("lambda_guess required for fixed hyperparams.")
        lambda_star = lambda_guess
        logging.info(f"Fixed lambda {lambda_star}...")

    if not fix_hyper:
        xi_0 = xi_guess or np.log(0.05)

        def kiv2_obj(xi: float):
            return kiv2_loss(df, np.exp(lambda_star), np.exp(xi))

        logging.info("Optimize xi...")
        res = minimize(kiv2_obj, x0=xi_0, method='BFGS')
        if not res.success:
            logging.info("KIV2 minimization did not succeed.")
        xi_star = res.x
        logging.info(f"Optimal xi {xi_star}...")
    else:
        if xi_guess is None:
            raise ValueError("xi_guess required for fixed hyperparams.")
        xi_star = xi_guess
        logging.info(f"Fixed xi {xi_star}...")

    logging.info("Predict treatment effect...")
    ystar = kiv_pred(df, np.exp(lambda_star), np.exp(xi_star), 3)
    return xstar, ystar
