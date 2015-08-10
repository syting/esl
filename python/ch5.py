#!/usr/local/bin/python

import eslii
import matplotlib.pyplot as plt
import numpy as np
import pandas
import random
import splines

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import LSQUnivariateSpline
from sklearn.linear_model import LogisticRegression


def figure_5_3():
    """ Reproduces figure 5.3 in ESLii displaying the pointwise variance curves
    for different spline bases
    """
    def pv(D):
        cv = np.linalg.inv(np.dot(D.transpose(), D))
        return [np.dot(D[i, :].transpose(), np.dot(cv, D[i, :]))
                for i in range(D.shape[0])]
    n = 50
    X = np.sort([random.random() for i in range(n)])
    l_X = np.ndarray((n, 2))
    l_X[:, 0] = np.ones(n)
    l_X[:, 1] = X
    l_pv = pv(l_X)
    c_X = np.ndarray((n, 4))
    c_X[:, 0:2] = l_X
    c_X[:, 2] = [x**2 for x in X]
    c_X[:, 3] = [x**3 for x in X]
    c_pv = pv(c_X)
    cs_X = np.ndarray((n, 6))
    cs_X[:, 0:4] = c_X
    cs_X[:, 4] = [max(0, (x - .33)**3) for x in X]
    cs_X[:, 5] = [max(0, (x - .66)**3) for x in X]
    cs_pv = pv(cs_X)

    ncs_X = splines.ns_basis(X, knots=[0.1, 0.26, 0.42, 0.58, 0.74, 0.9],
                             intercept=True)
    ncs_pv = pv(ncs_X)
    plt.plot(X, l_pv)
    plt.plot(X, c_pv)
    plt.plot(X, cs_pv)
    plt.plot(X, ncs_pv)


def figure_5_4():
    """Reproduces figure 5.4 in ESLii displaying the fitted natural spline for
    each term
    """
    data = eslii.read_sa_heart_data()
    data.drop(["adiposity", "typea", "alcohol"], axis=1, inplace=True)
    y = data["chd"]
    X = data.drop("chd", axis=1)
    X["famhist"] = pandas.get_dummies(X["famhist"])["Present"]
    N = np.ndarray((X.shape[0], 21))

    q = [0, 25, 50, 75, 100]
    N[:, 0:4] = splines.ns_basis(X["sbp"],
                                 knots=np.percentile(X["sbp"], q),
                                 intercept=False)
    N[:, 4:8] = splines.ns_basis(X["tobacco"],
                                 knots=np.percentile(X["tobacco"], q),
                                 intercept=False)
    N[:, 8:12] = splines.ns_basis(X["ldl"],
                                  knots=np.percentile(X["ldl"], q),
                                  intercept=False)
    N[:, 12] = X["famhist"]
    N[:, 13:17] = splines.ns_basis(X["obesity"],
                                   knots=np.percentile(X["obesity"], q),
                                   intercept=False)
    N[:, 17:21] = splines.ns_basis(X["age"],
                                   knots=np.percentile(X["age"], q),
                                   intercept=False)

    lr = LogisticRegression(C=1e50).fit(N, y)
    N -= N.mean(axis=0)

    fig = plt.figure()
    fig.add_subplot(321).scatter(X["sbp"], np.dot(N[:, 0:4], lr.coef_[0][0:4]))
    fig.add_subplot(322).scatter(X["tobacco"], np.dot(N[:, 4:8], lr.coef_[0][4:8]))
    fig.add_subplot(323).scatter(X["ldl"], np.dot(N[:, 8:12], lr.coef_[0][8:12]))
    fig.add_subplot(324).scatter(X["famhist"], np.dot(N[:, 12:13], lr.coef_[0][12:13]))
    fig.add_subplot(325).scatter(X["obesity"], np.dot(N[:, 13:17], lr.coef_[0][13:17]))
    fig.add_subplot(326).scatter(X["age"], np.dot(N[:, 17:21], lr.coef_[0][17:21]))
    plt.show()

def figure_5_5():
    """Reproduces figure 5.5 in ESLii displaying the results of fitting a spline
    to the phoneme classification result
    """
    phoneme = eslii.read_phoneme_data()
    aa = phoneme[phoneme['g'] == 'aa']
    aa_train = aa[map(lambda s: s.find("train") == 0, aa['speaker'])]
    aa_train = aa_train.reset_index()
    aa_test = aa[map(lambda s: s.find("test") == 0, aa['speaker'])]
    aa_test = aa_test.reset_index()
    ao = phoneme[phoneme['g'] == 'ao']
    ao_train = ao[map(lambda s: s.find("train") == 0, ao['speaker'])]
    ao_train = ao_train.reset_index()
    ao_test = ao[map(lambda s: s.find("test") == 0, ao['speaker'])]
    ao_test = ao_test.reset_index()

    # Print some examples of the data
    fit = plt.figure()
    for i in range(15):
        fit.add_subplot(211).plot(range(1, 257), aa_train.ix[i][1:257], c='green')
        fit.add_subplot(211).plot(range(1, 257), ao_train.ix[i][1:257], c='red')

    # Separate out train and test data/labels
    train_X = np.concatenate((aa_train[aa_train.columns[1:257]],
                              ao_train[ao_train.columns[1:257]]))
    train_y = [0 if i < aa_train.shape[0]
               else 1 for i in range(train_X.shape[0])]
    test_X = np.concatenate((aa_test[aa_test.columns[1:257]],
                             ao_test[ao_test.columns[1:257]]))
    test_y = [0 if i < aa_test.shape[0] else 1 for i in range(test_X.shape[0])]

    # Train raw classifier
    lr = LogisticRegression(C=1e50).fit(train_X, train_y)
    print "Raw errors: {:.2f} {:.2f}".format(1 - lr.score(train_X, train_y),
                                             1 - lr.score(test_X, test_y))

    # Train regularized classifier
    N = splines.ns_basis(range(1, 257), [1, 21, 42, 64, 85, 106, 128, 149, 170,
                                         192, 213, 234, 256])
    lr2 = LogisticRegression(C=1e50).fit(np.dot(train_X, N), train_y)
    print "Reg errors: {:.2f} {:.2f}".format(1 - lr2.score(np.dot(train_X, N),
                                                           train_y),
                                             1 - lr2.score(np.dot(test_X, N),
                                                           test_y))

    fit.add_subplot(212).plot(range(1, 257), lr.coef_[0],
                              range(1, 257), np.dot(N, lr2.coef_[0]))


def figure_5_6():
    """Reproduces figure 5.6 in ESLii fitting separate smoothing splines to
    male and female bone-density vs. age data
    """
    bone = eslii.read_bone_data()
    male = bone[bone["gender"] == "male"].sort("age")
    female = bone[bone["gender"] == "female"].sort("age")

    plt.scatter(male["age"], male["spnbmd"], color="blue")
    plt.scatter(female["age"], female["spnbmd"], color="red")

    ms = UnivariateSpline(male["age"], male["spnbmd"], s=.37)
    fs = UnivariateSpline(female["age"], female["spnbmd"], s=.313)

    plt.plot(male["age"], ms(male["age"]), color="blue")
    plt.plot(female["age"], fs(female["age"]), color="red")

def figure_5_9():
    """Reproduces figure 5.9 in ESLii displaying EPE and CV curves for different
    realizations of fitting a spline to a nonlinear function
    TODO: Finish implementation. Need to find or build methods for building
    design matrix for natural cubic spline.
    """
    n = 100
    X = np.sort([random.random() for i in range(n)])
    f = [sin(12*(x + 0.2)/(x + 0.2)) for x in X]
    y = [f_x + randn() for f_x in f]

    # Find K from the Reinsch representation of S_lambda and use to generate

