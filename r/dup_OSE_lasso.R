#
# Duplicate part of Figure 3.7 (the lasso regularization results) from Chapter 3 of the book ESLII
#
# Written by:
# --
# John L. Weatherwax
#-----

source("load_prostate_data.R")

PD = load_prostate_data(globalScale=FALSE, trainingScale=TRUE, responseScale=TRUE)
XTraining = PD[[1]]
XTesting = PD[[2]]

p = dim(XTraining)[2] - 1 # the last column is the response
nSamples = dim(XTraining)[1]

library(glmnet)

# alpha = 1 => lasso
fit = glmnet(as.matrix(XTraining[,1:p]), XTraining[,p+1], family="gaussian", alpha=1)

postscript("dup_fig_3_10.eps", onefile=FALSE, horizontal=FALSE)
plot(fit)
dev.off()

# do cross-validation to get the optimal value of lambda
cvob = cv.glmnet(as.matrix(XTraining[, 1:p]), XTraining[, p+1], family="gaussian", alpha=1)

postscript("dup_fig_3_7_lasso.eps", onefile=FALSE, horizontal=FALSE)
plot(cvob)
dev.off()

# get the optimal value of lambda:
lambdaOptimal = cvob$lambda.lse

# refit with this optimal value of lambda:
fitOpt = glmnet(as.matrix(XTraining[, 1:p]), XTraining[, p+1], family="gaussian",
                lambda = lambdaOptimal, alpha=1)
print(coef(fitOpt), digit=3)

# predict the testing data using this value of lambda:
yPredict = predict(fit, newx=as.matrix(XTesting[,1:p]), s=lambdaOptimal)
NTest = dim(XTesting[, 1:p])[1]
print(mean((XTesting[, p+1] - yPredict)^2), digit=3)
print(sqrt(var((XTesting[, p+1] - yPredict)^2)/NTest), digit=3)
