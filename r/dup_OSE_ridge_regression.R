#
# Duplicate part of Figure 3.7 (the ridge regression results) from Chapter 3 of the book ESLII
#
# Written by:
# --
# John L. Weatherwax
#-----

source("cv_ridge.R")
source("load_prostate_data.R")
source("one_standard_error_rule.R")
source("df_ridge.R")
source("opt_lambda_ridge.R")

PD = load_prostate_data(globalScale=FALSE, trainingScale=FALSE, responseScale=FALSE)
XTraining = PD[[1]]
XTesting = PD[[2]]

p = dim(XTraining)[2] - 1 # the last column is the response
nSamples = PD[[2]]

# Do ridge-regression cross validation
allLambdas = opt_lambda_ridge(XTraining[,1:p], multiple=2)
print(allLambdas)
numberOfLambdas = length(allLambdas)

for (li in 1:numberOfLambdas) {
    res = cv_ridge(allLambdas[li], XTraining, numberOfCV=10)

    # To get a *single* value for the degrees of freedom use the call on the entire training data set
    # We could also take the mean/median of this computed from the cross validation samples

    dof = df_ridge(allLambdas[li], as.matrix(XTraining[,1:p]))
    if (li == 1) {
        complexityParam = dof
        cvResults = res$cvraw
    } else {
        complexityParam = cbind(complexityParam, dof) 
        cvResults = cbind(cvResults, res$cvraw)
    }
}

# flip the order of the results:
inds = rev(1:numberOfLambdas)
allLambdas = allLambdas[inds]
complexityParam = complexityParam[inds]
cvResults = cvResults[,inds]

# group all cvResults by lambda (the complexity parameter) and compute statistics:
OSE = one_standard_error_rule(complexityParam, cvResults)
complexVValue = OSE[[1]]
complexIndex = OSE[[2]]
oseHValue = OSE[[3]]
means = OSE[[4]]
stds = OSE[[5]]

library(gplots) # plotCI, plotmeans found here

saveFig = TRUE
if (saveFig) postscript("dup_fig_3_7_ridge_regression.eps", onefile=FALSE, horizontal=FALSE)
plotCI(x=complexityParam, y=means, uiw=0.5*stds, xlim=c(0,8), ylim=c(0.3,1.8), col="black",
       barcol="blue", lwd=1, type="l", xlab="degrees of freedom", ylab="expected squared prediction error (ESPE)")
abline(h=oseHValue, lty=2)
abline(v=complexVValue, lty=2)
if (saveFig) dev.off()

# pick the best model, retrain over the entire data set, and predict on the testing data set:
bestLambda = allLambdas[complexIndex]

DCV = XTraining
lpsa = DCV[,p+1]
DCV$lpsa = NULL

# Standardize the predictors and demean the response
DCV = scale(DCV)
lpsaMean = mean(lpsa)
lpsa = lpsa - lpsaMean
DCVb = cbind(DCV, lpsa)
DCVf = data.frame(DCVb)

means = attr(DCV, "scaled:center")
stds = attr(DCV, "scaled:scale")

# apply the computed scaling based on the training data to the testing data:
DCVTest = XTesting
lpsaTest = DCVTest[, p+1] - lpsaMean
DCVTest$lpsa = NULL
DCVTest = t(apply(DCVTest, 1, '-', means))
DCVTest = t(apply(DCVTest, 1, '/', stds))
DCVTestb = cbind(DCVTest, lpsaTest)
DCVTestf = data.frame(DCVTestb)
names(DCVTestf)[p+1] = "lpsa"

# fit this linear model and compute the expected prediction error (EPE) using these features
DM = as.matrix(DCV)
M = ginv(t(DM) %*% DM + bestLambda*diag(p)) %*% t(DM)
betaHat = M %*% as.matrix(DCVf[, p+1])

print(lpsaMean, digits=4)
print(betaHat, digits=3)

DTest = as.matrix(DCVTest)
pdt = DTest %*% betaHat # get predictions on the test set
mErr = mean((lpsaTest - pdt)^2)
print(mErr)
NTest = length(lpsaTest)
sErr = sqrt(var((lpsaTest - pdt)^2)/NTest)
print(sErr)
