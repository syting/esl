load_prostate_data <- function(globalScale=FALSE, trainingScale=TRUE, responseScale=FALSE) {
    #
    # R code to load in the prostate data set from the book ESLII
    #
    # Output:
    #
    # res: list of data frames XT
    #
    # Written by:
    # _
    # John L. Weatherwax
    #
    #-----

    X = read.table("../data/prostate.data")

    # Based on the comments in the file prostate.info we try to scale ALL
    # features so that they have mean one and standard deviation of one.
    #
    # This would seem to violate the idea of separating training and
    # testing since the testing features would contribute to the mean and
    # variance used in scaling

    if (globalScale) {
        if (responseScale) {
            lpsa = X$lpsa - mean(X$lpsa)
        } else {
            lpsa = X$lpsa
        }
        train = X$train
        X$lpsa = NULL
        X$train = NULL
        X = scale(X, TRUE, TRUE)
        Xf = data.frame(X)
        Xf$lpsa = lpsa
        Xf$train = train
        X = Xf
        rm(Xf)
        rm(lpsa)
    }

    # separate into training/testing sets
    #
    XTraining = subset(X, train)
    XTraining$train = NULL # remove the training/testing column
    p = dim(XTraining)[2] - 1
    XTesting = subset(X, train==FALSE)
    XTesting$train = NULL

    # Sometimes data is processed and stored in a certain order. When doing cross validation
    # on such data sets we don't want to bias our results if we grab the first or the last samples.
    # Thus we randomize the order of the rows in the Training data frame to make sure that each 
    # cross validation training/testing set is as random as possible.

    if (FALSE) {
        nSamples = dim(XTraining)[1]
        inds = sample(1:nSamples, nSamples)
        XTraining = XTraining[inds,]
    }

    # In reality we have to estimate everything based on the training data only.
    # Here we estimate predictor statistics using the training set and then scale
    # the testing set by the same statistics

    if (trainingScale) {
        X = XTraining
        if (responseScale) {
            meanLpsa = mean(X$lpsa)
            lpsa = X$lpsa - meanLpsa
        } else {
            lpsa = X$lpsa
        }
        X$lpsa = NULL
        X = scale(X, TRUE, TRUE)
        means = attr(X, "scaled:center")
        stds = attr(X, "scaled:scale")
        Xf = data.frame(X)
        Xf$lpsa = lpsa
        XTraining = Xf

        # Scale the testing predictors by the same amounts
        DCVTest = XTesting
        if (responseScale) {
            lpsaTest = DCVTest$lpsa - meanLpsa
        } else {
            lpsaTest = DCVTest$lpsa # in physical units (not mean adjusted)
        }
        DCVTest$lpsa = NULL
        DCVTest = t(apply(DCVTest, 1, '-', means))
        DCVTest = t(apply(DCVTest, 1, '/', stds))
        DCVTestb = cbind(DCVTest, lpsaTest) # append back on the response
        DCVTestf = data.frame(DCVTestb) # a data frame containing all scaled variables of interest
        names(DCVTestf)[p+1] = "lpsa" # fix the name of the response
        XTesting = DCVTestf
    }

    return (list(XTraining, XTesting))
}
