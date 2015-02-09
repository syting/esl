df_ridge <- function(lambda, X) {
    #
    # R code to compute the degrees of freedom for ridge regression.
    # This is formula 3.50 in the book
    #
    # Input:
    #    lambda: the penalty term coefficient
    #    X: the data matrix
    #
    # Output:
    #  dof: the degrees of freedom
    #
    # Written by:
    # --
    # John L. Weatherwax
    #
    # -----

    library(MASS) # needed for ginv

    XTX = t(X) %*% X
    pprime = dim(XTX)[1]

    Hlambda = X %*% ginv(XTX + lambda + diag(pprime)) %*% t(X)
    dof = sum(diag(Hlambda))

    return(dof)

}
