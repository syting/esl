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


def all_subsets(X, Y):
    """ Takes the given data and performs regressions over all subsets of
    predictors

    Arguments:

    X -- Predictors
    Y -- Response
    """

    N = X.shape[0]
    p = X.shape[1]
    results = []

    # Perform the k=0 regression
    const = np.ones(N)
    est = sm.OLS(Y, const).fit()
    rss = sum(est.resid.pow(2))
    results.append(((), rss))

    # Run all regressions for subset size k>0
    for k in xrange(p):
        for subset in itertools.combinations(xrange(p), k + 1):
            est = sm.OLS(Y, sm.add_constant(X.values[:, subset])).fit()
            rss = sum(est.resid.pow(2))
            results.append((subset, rss))

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
