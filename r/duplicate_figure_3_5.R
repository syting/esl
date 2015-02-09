# Duplicate Figure 3.5 (all possible subset models) from the book

# Load the standardized data
source('load_prostate_data.R')
res = load_prostate_data(globalScale=TRUE, trainingScale=FALSE, responseScale=FALSE)
XTraining = res[[1]]
XTesting = res[[2]]

nrow = dim(XTraining)[1]
p = dim(XTraining)[2] - 1

# do the k=0 (no features) subset first:
mk = lm(lpsa ~ +1, data=XTraining)
rss = sum(mk$residuals^2)
xPlot = c(0); yPlot = c(rss)

# do all remaining 0<k subsets:
for (k in 1:p) {  # k=0 needs to be done outside of this loop
    allPosSubsetsSizeK = combn(p, k)  # get all possible subsets of size k
    numOfSubsets = dim(allPosSubsetsSizeK)[2]

    for (si in 1:numOfSubsets) {
        featIndices = allPosSubsetsSizeK[, si]
        featNames = as.vector(names(Df))[featIndices]

        # construct a formula needed by lm:
        form = "lpsa ~ "
        for (ki in 1:k) {
            if (ki == 1) {
                form = paste(form, featNames[ki], sep=" ")
            } else {
                form = paste(form, featNames[ki], sep="+")
            }
        }

        # fit this linear model and compute the RSS using these features:
        mk = lm(formula=form, data=Df)
        rss = sum(mk$residuals^2)
        xPlot = c(xPlot, k)
        yPlot = c(yPlot, rss)
    }
}

plot(xPlot, yPlot, xlab="Subset Size K", ylab="Residual Sum-of-Squares", ylim=c(0,100),xlim=c(0,8))

xMinPlot = xPlot[1]; yMinPlot = yPlot[1]
for (ki in 1:p) {
    inds = (xPlot == ki)
    rmin = min(yPlot[inds])
    xMinPlot = c(xMinPlot, ki); yMinPlot = c(yMinPlot, rmin)
}
lines(xMinPlot,yMinPlot)

#dev.off()
