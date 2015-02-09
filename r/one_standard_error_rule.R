one_standard_error_rule <- function(complexityParam, cvResults) {
    # Returns the least complex model whose mean squared prediction error is within one stdev 
    # of the best model
    # 
    # Input:
    #   complexityParam = the sampled complexity parameter
    #   cvResults = matrix where each column is a different value of the complexity parameter
    #       and each row is the square prediction error (SPE) of a different sample
    #
    # Output:
    #   optCP = the optimal complexity parameter
    #
    # Written by:
    # --
    # John L. Weatherwax
    #-----

    # this code is modeled after the code in "glmnet/cvelnet.R"
    N = dim(cvResults)[1]

    # compute the complexity based mean and standard deviations:
    means = apply(cvResults, 2, mean)
    stds = sqrt(apply(cvResults, 2, var)/N)

    # find the smallest ESPE:
    minIndex = which.min(means)

    # compute the CI around this point:
    ciw = stds[minIndex]

    # add a width of one std to the min point:
    maxUncertInMin = means[minIndex] + 0.5*ciw

    # find the mean that is nearest to this value:
    complexityIndex = which.min(abs(means - maxUncertInMin))
    complexityValue = complexityParam[complexityIndex]

    # package everything to send out:
    res = list(complexityValue, complexityIndex, maxUncertInMin, means, stds)

    return(res)

}
