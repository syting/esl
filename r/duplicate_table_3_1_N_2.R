# Duplicates Tables 3.1 and 3.2 from ESLII
#
# Written by:
# -
# John L. Weatherwax
#
#-----

source('load_prostate_data.R')
res = load_prostate_data(globalScale=TRUE, trainingScale=FALSE, responseScale=FALSE)
XTraining = res[[1]]
XTesting = res[[2]]

nrow = dim(XTraining)[1]
p = dim(XTraining)[2] - 1 # the last column is the response

D = XTraining[,1:p] # get the predictor data

# This gives the raw data for Table 3.2 from the book:
print(cor(D),digits=3)

library(xtable)
xtable(cor(D), caption="Duplication of the values from Table 3.1 from the book", digits=3)

# Duplicate Table 3.2 from the book

# Append a column of ones:
Dp = cbind(matrix(1, nrow, 1), as.matrix(D))
lpsa = XTraining[,p+1]

library(MASS)
betaHat = ginv(t(Dp) %*% Dp) %*% t(Dp) %*% as.matrix(lpsa)

# this is basically the first column in Table 3.2:
print('first column: beta estimates')
print(betaHat, digits=2)

# make predictions based on these estimated beta coefficients:
yhat = Dp %*% betaHat

# estimate the variance:
sigmaHat = sum((lpsa - yhat)^2)/(nrow - p - 1)

# calculate the covariance of betahat:
covarBetaHat = sigmaHat * ginv(t(Dp) %*% Dp)

# calculate the standard deviations of betahat:
stdBetaHat = sqrt(diag(covarBetaHat))

# this is basically the second column of Table 3.2:
print('second column: beta standard errors')
print(as.matrix(stdBetaHat), digits=2)

# compute the z-scores:
z = betaHat/stdBetaHat

# this is basically the third column in Table 3.2:
print('third column: beta z-scores')
print(z, digits=2)

# reproduce/verify the above results using R's "lm" function:
Db = cbind(D, lpsa)
Df = data.frame(Db)
m0 = lm(lpsa ~ lcavol + lweight + age + lbph + svi + lcp + gleason + pgg45, data=Df)
print('using the R lm function')
print(summary(m0), digits=2)

# display the results we get:
F = data.frame(Term=names(coefficients(m0)), Coefficients=betaHat, StdError=stdBetaHat, ZScore=z)
xtable(F, caption="Duplicated results for the books Table 3.2", digits=2)

# Run this full linear model on the Testing data so taht we can fill in the two
# lower spots in the "LS" column in Table 3.2
#

pdt = predict(m0, newdata=XTesting, interval="prediction")[,1] # get predictions on the test set
lpsaTest = XTesting$lpsa
NTest = length(lpsaTest)
mErr = mean((lpsaTest - pdt)^2)
print(mErr)
sErr = sqrt(var((lpsaTest - pdt)^2)/NTest)
print(sErr)

