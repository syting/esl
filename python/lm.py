import itertools
import numpy as np
import statsmodels.api as sm
from pandas import DataFrame


def standardize(df, cols=None):
    """Takes the given DataFrame, demeans some of the columns, standardizes them
    to have unit variance, and returns the result.

    Arguments:

    df -- The DataFrame to be standardized

    Keyword Arguments:

    cols -- The names of the columns to be standardized. If not given then all
    columns will be standardized
    """

    assert isinstance(df, DataFrame)

    if cols is None:
        cols = df.columns

    df_subset = df[cols]
    return (df_subset - df_subset.mean())/df_subset.std()


def all_subsets_k(X, Y, k):
    """Takes the given data and performs regressions over all subsets of the
    predictors containing k variables
    """
    N = X.shape[0]
    p = X.shape[1]
    assert(0 <= k and k <= p)
    results = []

    if k == 0:
        # Perform the k=0 regression
        const = np.ones(N)
        est = sm.OLS(Y, const).fit()
        rss = sum(est.resid.pow(2))
        results.append(((), rss))
    else:
        for subset in itertools.combinations(xrange(p), k):
            est = sm.OLS(Y, sm.add_constant(X.values[:, subset])).fit()
            rss = sum(est.resid.pow(2))
            results.append((subset, rss))
    return results


def best_subset_k(X, Y, k):
    """Returns the best set of indices of size k and RSS when performing OLS
    over those predictors
    """

    return min(all_subsets_k(X, Y, k), key=lambda x: x[1])


def all_subsets(X, Y):
    """ Takes the given data and performs regressions over all subsets of
    predictors

    Arguments:

    X -- Predictors
    Y -- Response
    """

    p = X.shape[1]
    results = []

    for k in xrange(p + 1):
        results.extend(all_subsets_k(X, Y, k))

    return results


def ridge_regression(X, Y, Lambda):
    """ Takes the given data and performs ridge regression parameterized by the
    given lambda

    Arguments:

    X -- Predictors
    Y -- Response
    Lambda -- lambda value to use in the ridge regression
    """
    pass
