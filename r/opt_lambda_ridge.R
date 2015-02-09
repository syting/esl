opt_lambda_ridge <- function(X, multiple=1) {
    #
    # Returns the "optimal" set of lambda for nice looking ridge regression plots. See the
    # notes that accompanies the book Elements of Statistical Learning
    #
    # Input:
    #
    #   X = matrix or DataFrame of features (do not prepend a constant column)
    #
    # Output:
    #
    #   lambdas = vector of optimal lambda (chosen to make the DFRidge(lambda) equal to the
    #       integers 1:p
    #
    # Written by:
    # --
    # John L. Weatherwax
    #
    #-----

    s = svd(X)
    dj = s$d    # the diagonal elements of D in decreasing order
    p = dim(X)[2]

    # Do df = p first (this is the lambda value when the problem is unconstrained)
    lambdas = 0

    # Do all other values next:
    kRange = seq(p-1, 1, by=(-1/multiple))
    for (ki in 1:length(kRange)) {
        # solve for lambda in (via Newton iterations):
        #   k = \sum_{i=1}^p \frac{d_j^2}{d_j^2 + \lambda}
        k = kRange[ki]

        # initial guess at the root
        if (ki == 1) {
            xn = 0.0
        } else {
            xn = xnp1 # use the oldest previously computed root
        }

        f = sum(dj^2/(dj^2 + xn)) - k # do the first update by hand
        fp = -sum(dj^2/(dj^2 + xn)^2)
        xnp1 = xn - f/fp

        # Find the root within tolerance .001
        while (abs(xn - xnp1)/abs(xn) > 10^(-3)) {
            xn = xnp1
            f = sum(dj^2/(dj^2 + xn)) - k
            fp = -sum(dj^2/(dj^2 + xn)^2)
            xnp1 = xn - f/fp
        }

        lambdas = c(lambdas, xnp1)
    }
    # flip the order of the lambdas:
    lambdas = lambdas[rev(1:length(lambdas))]
    return(lambdas)
}
