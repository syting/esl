#!/usr/local/bin/python

import eslii
import matplotlib.pyplot as plt
import pandas
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.qda import QDA


def figure_4_4_and_4_8(i, j):
    """Reproduces figure 4.4 and 4_8 in ESLii displaying canonical coordinates
    for the vowel data. Here i and j specify which canonical coordinates to show
    where i and j range from 1 to 10.
    """
    assert(1 <= i and 1 <= j and i < j and i <= 9 and j <= 10)
    vowels = eslii.read_vowel_data()
    X = vowels[vowels.columns[1:]]
    y = vowels['y']
    rr_x = LDA().fit_transform(X, y)

    plt.scatter(rr_x[:, i - 1], rr_x[:, j - 1], c=y)


def table_4_1():
    """Reproduces table 4.1 in ESLii showing the training and test error rates
    for classifying vowels using different classification techniques. The
    sklearn implementation of logistic regression uses OvA instead of a true
    multinomial which likely accounts for the worse results
    """
    vowels_train = eslii.read_vowel_data()
    train_X = vowels_train[vowels_train.columns[1:]]
    train_y = vowels_train['y']
    vowels_test = eslii.read_vowel_data(train=False)
    test_X = vowels_test[vowels_test.columns[1:]]
    test_y = vowels_test['y']

    lda = LDA().fit(train_X, train_y)
    print "Linear discriminant analysis:  {:.2f} {:.2f}".format(
        1 - lda.score(train_X, train_y), 1 - lda.score(test_X, test_y))
    qda = QDA().fit(train_X, train_y)
    print "Quadratic discriminant analysis:  {:.2f} {:.2f}".format(
        1 - qda.score(train_X, train_y), 1 - qda.score(test_X, test_y))
    lr = LogisticRegression(C=1e30).fit(train_X, train_y)
    print "Logistic regression:  {:.2f} {:.2f}".format(
        1 - lr.score(train_X, train_y), 1 - lr.score(test_X, test_y))


def tables_4_2_and_4_3():
    """Reproduces table 4.2 and 4.3 in ESLii showing the results of a logistic
    regression fit to selected predictors of the South African heart disease
    data
    """
    data = eslii.read_sa_heart_data()
    data.drop([u"adiposity", u"typea"], axis=1, inplace=True)
    y = data["chd"]
    X = data.drop("chd", axis=1)
    X["famhist"] = pandas.get_dummies(X["famhist"])["Present"]
    lr = LogisticRegression(C=1e30).fit(X, y)
    print "(Intercept) {:.3f}".format(lr.intercept_[0])
    for (i, column) in enumerate(X.columns):
        print "{} {:.3f}".format(column, lr.coef_[0][i])

    print "\n"
    X.drop(["sbp", "obesity", "alcohol"], axis=1, inplace=True)
    lr = LogisticRegression(C=1e30).fit(X, y)
    print "(Intercept) {:.3f}".format(lr.intercept_[0])
    for (i, column) in enumerate(X.columns):
        print "{} {:.3f}".format(column, lr.coef_[0][i])


def figure_4_13():
    """Reproduces figure 4.13 in ESLii showing the coefficients of an
    L1-regularized logistic regression fit to the South African heart disease
    data as a function of the L1 length of beta
    TODO: this doesn't match
    """
    data = eslii.read_sa_heart_data()
    data.drop([u"adiposity", u"typea"], axis=1, inplace=True)
    y = data["chd"]
    X = data.drop("chd", axis=1)
    X["famhist"] = pandas.get_dummies(X["famhist"])["Present"]
    X = eslii.standardize_data(X, demeanCols=[])
    beta_norms = []
    coefs = {}
    for column in X.columns:
        coefs[column] = []
    alphas = [1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1,
              .5, 1.0, 10.0]
    for alpha in alphas:
        lr = LogisticRegression(penalty="l1", C=alpha).fit(X, y)
        beta_norms.append(sum(abs(lr.coef_[0])))
        for (i, column) in enumerate(X.columns):
            coefs[column].append(lr.coef_[0][i])

    for column in X.columns:
        plt.plot(beta_norms, coefs[column])
