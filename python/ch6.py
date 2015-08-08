#!/usr/local/bin/python

import eslii
import matplotlib.pyplot as plt
import numpy as np
import pyqt_fit.nonparam_regression as smooth

from kernel_regression import KernelRegression
from pyqt_fit import npr_methods
from sklearn.neighbors import KNeighborsRegressor, KernelDensity


def figures_6_1_and_6_3():
    """Reproduces figure 6.1 in ESLii showing different local regression fits
    to Y = sin(4X) + eps where X~U[0,1] and eps~N(0, 1/3) and overlays figure
    6.3 highlighting the bias of constant fits near the boundaries
    """

    # Produce the data an true function
    X = np.sort(np.random.rand(100, 1), axis=0)
    Y = np.sin(np.multiply(X, 4)).ravel()
    Y += np.random.normal(scale=.333, size=100)
    lin_x = np.linspace(0.0, 1.0, 100)
    f_x = np.sin(4*lin_x)

    # Plot the data
    plt.figure(1)
    plt.subplot(221)
    plt.scatter(X, Y)

    # Plot the true function f
    plt.plot(lin_x, f_x, label='f')

    # Plot a nearest-neighbor fit
    nbrs = KNeighborsRegressor(n_neighbors=30).fit(X, Y)
    fhat_x = nbrs.predict(lin_x.astype(np.ndarray).reshape((100, 1)))
    plt.plot(lin_x, fhat_x, label='nn')
    plt.legend(loc='best')

    # Plot a Nadaraya-Watson kernel-weighted fit using an Epanechnikov Kernel
    plt.subplot(222)
    plt.scatter(X, Y)
    plt.plot(lin_x, f_x, label='f')

    def epanechnikov_kernel(x0, x, gamma=1.0):
        d = abs(x0 - x)/gamma
        return (0.0 if 1 <= d else .75*(1 - d**2))
    kr = KernelRegression(kernel=epanechnikov_kernel, gamma=0.2).fit(X, Y)
    fhat_x = kr.predict(lin_x.astype(np.ndarray).reshape((100, 1)))
    plt.plot(lin_x, fhat_x, label='nw')
    plt.legend(loc='best')

    # Plot a lwlr fit
    plt.subplot(223)
    plt.scatter(X, Y)
    plt.plot(lin_x, f_x, label='f')
    k1 = smooth.NonParamRegression(X.reshape(100,),
                                   Y,
                                   bandwidth=0.1,
                                   method=npr_methods.LocalPolynomialKernel(
                                       q=1))
    k1.fit()
    fhat_x = k1(lin_x)
    plt.plot(lin_x, fhat_x, label='lwlr')
    plt.legend(loc='best')

    # Plot a lwqr fit
    plt.subplot(224)
    plt.scatter(X, Y)
    plt.plot(lin_x, f_x, label='f')
    k2 = smooth.NonParamRegression(X.reshape(100,),
                                   Y,
                                   bandwidth=0.3,
                                   method=npr_methods.LocalPolynomialKernel(
                                       q=2))
    k2.fit()
    fhat_x = k2(lin_x)
    plt.plot(lin_x, fhat_x, label='lwqr')
    plt.legend(loc='best')
    plt.show()


def figure_6_14():
    """Reproduces figure 6.14 in ESLii displaying a density estimate for sbp
    levels in chd/no-chd groups using a Gaussian kernel density estimate
    """
    sa = eslii.read_sa_heart_data()
    sbp = sa["sbp"]
    sbp_chd = sa[sa["chd"] == 1]["sbp"].copy()
    sbp_chd.sort()
    sbp_no_chd = sa[sa["chd"] == 0]["sbp"].copy()
    sbp_no_chd.sort()

    kde_chd = KernelDensity(kernel='gaussian', bandwidth=7.5).fit(
        sbp_chd.reshape(len(sbp_chd), 1))
    chd_log_dens = kde_chd.score_samples(sbp_chd.reshape((len(sbp_chd), 1)))
    plt.subplot(121)
    plt.plot(sbp_chd, np.exp(chd_log_dens), label="CHD")

    kde_no_chd = KernelDensity(kernel='gaussian', bandwidth=7.5).fit(
        sbp_no_chd.reshape(len(sbp_no_chd), 1))
    no_chd_log_dens = kde_no_chd.score_samples(
        sbp_no_chd.reshape((len(sbp_no_chd), 1)))
    plt.plot(sbp_no_chd, np.exp(no_chd_log_dens), label="no CHD")
    plt.legend(loc='best')

    sbp_range = np.linspace(min(sbp), max(sbp), 100).reshape((100, 1))
    chd_dens = np.exp(kde_chd.score_samples(sbp_range))
    no_chd_dens = np.exp(kde_no_chd.score_samples(sbp_range))
    p_chd = float(len(sbp_chd))/(len(sbp_chd) + len(sbp_no_chd))
    posterior_est = [p_chd * chd_dens[i] /
                     (p_chd * chd_dens[i] + (1 - p_chd) * no_chd_dens[i])
                     for i in range(len(sbp_range))]
    plt.subplot(122)
    plt.plot(sbp_range, posterior_est)
    plt.show()
