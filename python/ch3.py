#!/usr/local/bin/python

import eslii
import lm
import pandas
import matplotlib
import numpy as np
import scipy.optimize
from sklearn import cross_validation
from sklearn import cross_decomposition
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, lasso_path, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

DATA_DIR = "../data"


def table_3_1():
    """Reproduces Table 3.1, returning the result as a DataFrame"""

    df = eslii.read_prostate_data()
    train_predictors = df[df['train'] == 'T'][df.columns[:-2]]
    return train_predictors.corr()


def table_3_2():
    """Reproduces Table 3.2 as a statsmodels.iolib.summary.Summary. There is a
    slight difference because here we standardize the predictors after
    splitting into training/test data
    """

    # Get the standardized training data
    df = eslii.read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = eslii.standardize_data(train_data[train_data.columns[:-1]])
    train_predictors = sm.add_constant(train_predictors)  # Have to do manually
    train_response = train_data['lpsa']

    # Run OLS and return the result
    est = sm.OLS(train_response, train_predictors).fit()
    return est.summary()


def figure_3_5():
    """Reproduces Figure 3.5"""
    df = eslii.read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = eslii.standardize_data(train_data[train_data.columns[:-1]])
    train_response = train_data['lpsa']

    # Get all regressions on subsets of indices
    all_subsets_data = lm.all_subsets(train_predictors, train_response)
    x_data = [len(ss) for (ss, _) in all_subsets_data]
    y_data = [rss for (_, rss) in all_subsets_data]
    matplotlib.pyplot.scatter(x_data, y_data)


def get_all_subsets_errors():
    """ Performs 10-fold cross-validation on all-subset regression over the
    prostate data and returns MSPE and CI for effective degrees-of-freedom btw
    0 and p
    """
    df = eslii.read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = eslii.standardize_data(train_data[train_data.columns[:-1]])
    train_response = train_data['lpsa']

    subset_sizes = range(1, train_predictors.shape[1] + 1)
    lr = LinearRegression(fit_intercept=True)
    errors = [(1.45, .547)]
    for size in subset_sizes:
        model_err = []
        for fold in cross_validation.KFold(len(train_data), 10, shuffle=True):
            pred_idxs = lm.best_subset_k(pandas.DataFrame(train_predictors.as_matrix()[fold[0]]),
                                         pandas.Series(train_response.as_matrix()[fold[0]]),
                                         size)[0]
            model = lr.fit(train_predictors.as_matrix()[fold[0]][:, pred_idxs],
                           train_response.as_matrix()[fold[0]])
            model_err.append(mean_squared_error(model.predict(train_predictors.as_matrix()[fold[1]][:, pred_idxs]),
                                         train_response.as_matrix()[fold[1]]))
        errors.append((np.mean(model_err), np.std(model_err)))
    matplotlib.pyplot.errorbar(range(0, len(subset_sizes) + 1),
                               [x[0] for x in errors],
                               yerr=[x[1]/2 for x in errors],
                               ecolor='b', color='m')


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
    lambdas.append(scipy.optimize.newton(lambda x: f(x) - .001, lambdas[-1],
                                         fprime=fprime))
    lambdas.reverse()
    return lambdas


def get_ridge_errors():
    """ Performs 10-fold cross-validation on ridge regression over the prostate
    data and returns MSPE and CI for effective degrees-of-freedom btw 0 and p
    """
    df = eslii.read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = eslii.standardize_data(train_data[train_data.columns[:-1]])
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
    matplotlib.pyplot.errorbar(range(0, len(lambdas)),
                               [x[0] for x in errors],
                               yerr=[x[1]/2 for x in errors],
                               ecolor='b', color='m')

def get_lasso_errors():
    """ Performs 10-fold cross-validation on lasso regression over the prostate
    data and returns MSPE and CI for degrees-of-freedom btw 0 and p
    """
    df = eslii.read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = eslii.standardize_data(train_data[train_data.columns[:-1]])
    train_response = train_data['lpsa']

    alphas_lasso, coefs_lasso, _ = lasso_path(train_predictors,
                                              train_response,
                                              5e-3,
                                              fit_intercept=True)
    nv = [len(filter(lambda x : x != 0.0, coefs_lasso[:,i]))
          for i in range(0, coefs_lasso.shape[1])]
    nv.reverse()
    alphas = [alphas_lasso[nv.index(i)] for i in range(0, 9)]
    alphas.reverse()

    errors = []
    for a in alphas:
        clf = Lasso(alpha=a)
        model_err = -cross_validation.cross_val_score(clf, train_predictors,
                                                      train_response,
                                                      scoring='mean_squared_error',
                                                      cv=cross_validation.KFold(len(train_data), 10, shuffle=True))
        errors.append((model_err.mean(), model_err.std()))
    matplotlib.pyplot.errorbar(range(0, len(alphas)),
                               [x[0] for x in errors],
                               yerr=[x[1]/2 for x in errors],
                               ecolor='b', color='m')


def get_pcr_errors():
    """ Performs 10-fold cross-validation on principal components regression
    over the prostate data and returns the MSPE and CI numbers of directions btw
    0 and p
    """
    df = eslii.read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = eslii.standardize_data(train_data[train_data.columns[:-1]])
    train_response = train_data['lpsa']

    num_dirs = [i for i in range(1,train_predictors.shape[1] + 1)]
    errors = [(1.45, .547)]
    X = PCA().fit_transform(train_predictors)
    lr = LinearRegression()
    for num_dir in num_dirs:
        model_err = -cross_validation.cross_val_score(lr, X[:,:num_dir],
                                                      train_response,
                                                      scoring='mean_squared_error',
                                                      cv=cross_validation.KFold(len(train_data), 10, shuffle=True))
        errors.append((model_err.mean(), model_err.std()))
    matplotlib.pyplot.errorbar(range(0, len(num_dirs) + 1),
                               [x[0] for x in errors],
                               yerr=[x[1]/2 for x in errors],
                               ecolor='b', color='m')


def get_pls_errors():
    """ Performs 10-fold cross-validation on partial least squares regression
    over the prostate data and returns MSPE and CI for effective
    degrees-of-freedom btw 0 and p
    """
    df = eslii.read_prostate_data()
    train_data = df[df['train'] == 'T'][df.columns[:-1]]
    train_predictors = eslii.standardize_data(train_data[train_data.columns[:-1]])
    train_response = train_data['lpsa']

    num_dirs = [i for i in range(1,train_predictors.shape[1] + 1)]
    errors = [(1.45, .547)]
    for num_dir in num_dirs:
        clf = cross_decomposition.PLSRegression(n_components=num_dir)
        model_err = -cross_validation.cross_val_score(clf, train_predictors,
                                                      train_response,
                                                      scoring='mean_squared_error',
                                                      cv=cross_validation.KFold(len(train_data), 10, shuffle=True))
        errors.append((model_err.mean(), model_err.std()))
    matplotlib.pyplot.errorbar(range(0, len(num_dirs) + 1),
                               [x[0] for x in errors],
                               yerr=[x[1]/2 for x in errors],
                               ecolor='b', color='m')


def table_3_3():
    """Reproduces table 3.3 in ESLii.
    TODO: Requires concepts from Chapter 7
    """
    pass


def figure_3_8():
    """TODO: Reproduces figure 3.8 in ESLii displaying ridge regression
    coefficients as the effective degrees of freedom varies
    """
    pass


def figure_3_10():
    """TODO: Reproduces figure 3.10 in ESLii displaying lasso regression
    coefficients as the effective degrees of freedom varies
    """
    pass
