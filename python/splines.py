import numpy as np


def ns_basis(X, knots=None, intercept=False):
    """Returns a cubic natural spline design matrix for X with given knots. If
    knots are not specified then they are set to X
    """
    if knots is None:
        knots = np.sort(X)

    def dk(x, i):
        numer = (max(0, (x - knots[i])**3) - max(0, (x - knots[-1])**3))
        return numer/(knots[-1] - knots[i])

    n = len(X)
    p = len(knots) - (0 if intercept else 1)
    D = np.ndarray((n, p))

    i = 0
    if intercept:
        D[:, i] = np.ones(n)
        i += 1
    D[:, i] = X
    i += 1
    for j in range(p - 1):
        D[:, i] = [dk(x, j) - dk(x, p - 1) for x in X]
        i += 1
    return D
