load_vowel_data <- function(doScaling=FALSE, doRandomization=FALSE) {
    #
    # R code to load in the vowel data set from the book ESLII
    #
    # Output:
    #
    # res: list of data frames XT
    #
    # Written by:
    # John L. Weatherwax
    #-----

    # Get the training data
    XTrain = read.csv("../data/vowel.data", header=TRUE)

    # Delete the column named "row.names":
    XTrain$row.names = NULL

    # Extract the true classification for each datum
    labelsTrain = XTrain[,1]

    # Delete the column of classification labels:
    XTrain$y = NULL

    # We try to scale ALL features so that they have mean zero and unit variance
    if (doScaling) {
        XTrain = scale(XTrain, TRUE, TRUE)
        means = attr(XTrain, "scaled:center")
        stds = attr(XTrain, "scaled:scale")
        XTrain = data.frame(XTrain)
    }

    # Sometimes data is processed and stored on disk in a structured fashion. When doing cross-validation
    # we don't want ot bias our results so randomize the examples
    if (doRandomization) {
        nSamples = dim(XTrain)[1]
        inds = sample(1:nSamples, nSamples)
        XTrain = XTrain[inds,]
        labelsTrain = labelsTrain[inds]
    }

    # Get the testing data:
    XTest = read.csv("../data/vowels.test", header=TRUE)

    # Delete the column named "row.names":
    XTest$row.names = NULL

    # Extract the true classification for each datum
    labelsTest = XTest[,1]

    # Delete the column of classification labels
    XTest$y = NULL

    # Scale the testing data using the same transformation as was applied to the training data:
    if (doScaling) {
        XTest = t(apply(XTest, 1, '-', means))
        XTest = t(apply(XTest, 1, '/', stds))
    }

    return(list(XTrain, labelsTrain, XTest, labelsTest))
}
