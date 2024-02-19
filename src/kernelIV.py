from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


def kiv(Z, X, Ztilde, Ytilde, lam1, lam2, gamma):
    """ Kernel Instrumental Variables """
    n = Z.shape[0]
    m = Ztilde.shape[0]

    Kzztilde = rbf_kernel(Z, Ztilde, gamma)
    Kzz = rbf_kernel(Z, Z)
    Kxx = rbf_kernel(X, X)
    W = Kxx @ np.linalg.solve(Kzz + n * lam1 * np.eye(n), Kzztilde)
    alpha_hat = np.linalg.solve(W@W.T + m * lam2 * np.eye(m), W@Ytilde)

    return W, lambda xtilde: alpha_hat.T @ rbf_kernel(X, xtilde)