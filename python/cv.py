import numpy as np


def one_standard_error_rule(param, results):
    """ Takes in the result of a cross-validation run over the given param and
    applies the one standard error rule to determine the optimal parameter
    value for balancing prediction error with model complexity

    Arguments:

    param -- the sampled complexity parameter
    results -- DataFrame where each column is a different value of the
        complexity parameter and each row is the square prediction error of a
        different cross validation sample
    """

    # compute the complexity based mean and standard deviations:
    means = np.mean(results)
    stds = np.std(results)

    # find the smallest ESPE
    minIndex = means.argmin()

    # compute the confidence interval around this point:
    ciw = stds[minIndex]
    maxUncertInMin = means[minIndex] + .5*ciw

    # find the mean that is nearest to this value:
    complexityIndex = abs(means - maxUncertInMin).argmin()
    complexityValue = param[complexityIndex]

    return complexityValue, complexityIndex, maxUncertInMin, means, stds
