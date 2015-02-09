#!/usr/local/bin/python

import pandas as pd
import lm
import matplotlib
import numpy as np
import scipy.optimize
from sklearn import cross_validation
from sklearn import cross_decomposition
from sklearn.linear_model import Ridge
import statsmodels.api as sm

DATA_DIR = "../data"


def read_prostate_data():
    """Reads the prostate data from file into a DataFrame and returns this."""
    return pd.read_table(DATA_DIR + '/prostate.data', delim_whitespace=True)


def table_3_1():
    """Reproduces Table 3.1, returning the result as a DataFrame"""

    df = read_prostate_data()
    return df[df.columns[:-2]].corr()


def standardized_prostate_data():
    """Returns the prostate data as a DataFrame after standardizing the
    predictors to have zero mean and unit variance
    """

    df = read_prostate_data()
    predictor_cols = df.columns[:-2]
    df[predictor_cols] = lm.standardize(df, predictor_cols)
    return df


def table_3_2():
    """Reproduces Table 3.2 as a statsmodels.iolib.summary.Summary"""

    # Get the standardized training data
    df = standardized_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = train_data[train_data.columns[:-1]]
    train_predictors = sm.add_constant(train_predictors)  # Have to do manually
    train_response = train_data['lpsa']

    # Run OLS and return the result
    est = sm.OLS(train_response, train_predictors).fit()
    return est.summary()


def figure_3_5():
    """Reproduces Figure 3.5"""

    df = read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = train_data[train_data.columns[:-1]]
    train_response = train_data['lpsa']

    # Get all regressions on subsets of indices
    all_subsets_data = lm.all_subsets(train_predictors, train_response)
    x_data = [len(ss) for (ss, _) in all_subsets_data]
    y_data = [rss for (_, rss) in all_subsets_data]
    matplotlib.pyplot.scatter(x_data, y_data)


def opt_lambda_ridge(X, multiple=1):
    """ Takes in a DataFrame X and produces the set of lambda values
    corresponding to effective dof 1:p:multiple

    Arguments:

        X -- DataFrame representing the predictors
        multiple -- stepsize to take between 1 and p
    """
    dj = np.linalg.svd(X, compute_uv=False)
    p = X.shape[1]

    def f(l):
        return sum([d**2/(d**2 + l) for d in dj])

    def fprime(l):
        return -sum([d**2/(d**2 + l)**2 for d in dj])
    lambdas = [0]
    for k in range(p-multiple, 0, -multiple):
        r = scipy.optimize.newton(lambda x: f(x) - k, lambdas[-1],
                                  fprime=fprime)
        lambdas.append(r)
    lambdas.reverse()
    return lambdas


def get_ridge_errors():
    """ Performs 10-fold cross-validation on ridge regression over the prostate
    data and returns MSPE and CI for effective degrees-of-freedom btw 0 and p
    """
    df = standardized_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = train_data[train_data.columns[:-1]]
    train_response = train_data['lpsa']

    lambdas = opt_lambda_ridge(train_predictors)
    errors = []
    for l in lambdas:
        clf = Ridge(alpha=l)
        model_err = -cross_validation.cross_val_score(clf, train_predictors,
                                                      train_response,
                                                      scoring='mean_squared_error',
                                                      cv=cross_validation.KFold(len(train_data), 10, shuffle=True))
        errors.append((model_err.mean(), model_err.std()))
    matplotlib.pyplot.errorbar(range(1, len(lambdas) + 1),
                               [x[0] for x in errors],
                               yerr=[x[1]/2 for x in errors],
                               ecolor='b', color='m')

def get_pls_errors():
    """ Performs 10-fold cross-validation on ridge regression over the prostate
    data and returns MSPE and CI for effective degrees-of-freedom btw 0 and p
    """
    df = standardized_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = train_data[train_data.columns[:-1]]
    train_response = train_data['lpsa']

    num_dirs = [i for i in range(1,train_predictors.shape[1] + 1)]
    errors = []
    for num_dir in num_dirs:
        clf = cross_decomposition.PLSRegression(n_components=num_dir)
        model_err = -cross_validation.cross_val_score(clf, train_predictors,
                                                      train_response,
                                                      scoring='mean_squared_error',
                                                      cv=cross_validation.KFold(len(train_data), 10, shuffle=True))
        errors.append((model_err.mean(), model_err.std()))
    matplotlib.pyplot.errorbar(range(1, len(num_dirs) + 1),
                               [x[0] for x in errors],
                               yerr=[x[1]/2 for x in errors],
                               ecolor='b', color='m')
