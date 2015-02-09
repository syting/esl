cv_ridge <- function(lambda, D, numberOfCV) {
    #
    # Does cross-validation of the Ridge Regression Linear Model specified by the formula "form"
    #
    # Input:
    #   lambda = multiplier of the penalty term
    #   D = training DataFrame where the last column is the response variable
    #   numberOfCV = number of cross validations to do
    #
    # Output:
    #   MSPE: vector of length numberOfCV with components mean square prediction errors extracted
    #       from each of the numberOfCV test data sets
    #
    # Written by:
    # --
    # John L. Weatherwax
    #-----

    library(MASS) # needed for ginv

    nSamples = dim(D)[1]
    p = dim(D)[2] - 1 # the last column is the response

    # needed for the cross validation of (CV) loops:
    nCVTest = round(nSamples*(1/numberOfCV)) # each CV run will have this many test points
    nCVTrain = nSamples - nCVTest # each CV run will have this many training points

    for (cvi in 1:numberOfCV) {
        # Get the current test data and response
        testInds = (cvi - 1)*nCVTest + 1:nCVTest
        testInds = intersect(testInds, 1:nSamples)
        DCVTest = D[testInds,1:(p+1)]

        # select this cross-validation's section of data
        trainInds = setdiff(1:nSamples, testInds)
        DCV = D[trainInds, 1:(p+1)]

        # Get the response and delete it from the DataFrame
        responseName = names(DCV)[p+1]
        response = DCV[,p+1]
        DCV[[responseName]] = NULL

        # For ridge-regression we begin by standardizing the predictors and demeaning the response
        DCV = scale(DCV)
        responseMean = mean(response)
        response = response - responseMean
        DCVb = cbind(DCV, response) # append back on the response
        DCVf = data.frame(DCVb) # a data frame containing all scaled variables of interest

        # extract the centering and scaling info
        means = attr(DCV, "scaled:center")
        stds = attr(DCV, "scaled:scale")

        # apply the computed scaling based on the training data to the testing data:
        responseTest = DCVTest[,p+1] - responseMean
        DCVTest[[responseName]] = NULL
        DCVTest = t(apply(DCVTest, 1, '-', means))
        DCVTest = t(apply(DCVTest, 1, '/', stds))
        DCVTestb = cbind(DCVTest, responseTest) # reappend the response
        DCVTestf = data.frame(DCVTestb)
        names(DCVTestf)[p+1] = responseName

        # fit this linear model and compute the expected prediction error (EPE) using these features
        DM = as.matrix(DCV)
        M = ginv(t(DM) %*% DM + lambda*diag(p)) %*% t(DM)
        betaHat = M %*% as.matrix(DCVf[,p+1])

        DTest = as.matrix(DCVTest)
        pdt = DTest %*% betaHat # get demeaned predictions on the test set

        if (cvi == 1) {
            predmat = pdt
            y = responseTest
        } else {
            predmat = c(predmat, pdt)
            y = c(y, responseTest)
        }
    } # endfor cvi loop

    # this code is modeled after the code in "glmnet/cvelnet.R"
    N = length(y)
    cvraw = (y - predmat)^2
    cvm = mean(cvraw)
    cvsd = sqrt(var(cvraw)/N)
    l = list(cvraw=cvraw, cvm=cvm, cvsd=cvsd, name="Mean Squared Error")

    return(l)
}
