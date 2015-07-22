module Classification

using DataFrames, DataFramesUtil
using StatsBase

function indicator_matrix(y)
    num_categories = maximum(y)
    ind_mat = zeros(length(y), num_categories)
    for i in 1:length(y)
        ind_mat[i, y[i]] = 1
    end
    return ind_mat
end

type LRClassifier <: RegressionModel
    intercept::Vector
    B::Matrix
end

function StatsBase.fit(::Type{LRClassifier}, X::Matrix, y::Vector)
    Y = indicator_matrix(y)
    intercept = [mean(Y[:, i]) for i in 1:size(Y)[2]]
    for i in 1:size(Y)[2]
        Y[:, i] -= intercept[i]
    end
    B = (X'*X)^-1*X'*Y
    lrc = LRClassifier(intercept, B)
    return lrc
end

function StatsBase.predict(mm::LRClassifier, newX::Matrix)
    Y_hat = newX*mm.B .+ mm.intercept'
    return [indmax(Y_hat[i, :]) for i in 1:size(Y_hat)[1]]
end

type LDAClassifier
    priors::Vector
    centers::Matrix
    covariance::Matrix

    precision::Matrix
    const_term::Vector
end

function StatsBase.fit(::Type{LDAClassifier}, X::Matrix, y::Vector)
    n = size(y)[1]
    cats = unique(y)
    n_cats = length(cats)
    priors = [1.0*count(k -> k == i, y)/n for i in 1:n_cats]
    centers = zeros(size(X)[2], n_cats); 
    for i in 1:n_cats
        centers[:, i] = mean(X[y .== i, :], 1)
    end
    covariance = sum([(X[i, :]' - centers[:, y[i]])*(X[i, :]' - centers[:, y[i]])' for i in 1:size(X)[1]])/(n - n_cats)
    precision = covariance^-1
    const_term = [-.5*(centers[:, i]'*precision*centers[:, i])[1] + log(priors[i]) for i in 1:n_cats]

    return LDAClassifier(priors, centers, covariance, precision, const_term)
end

function StatsBase.predict(mm::LDAClassifier, newX::Matrix)
    discr_vals = newX*mm.precision*mm.centers .+ mm.const_term'
    return [indmax(discr_vals[i, :]) for i in 1:size(discr_vals)[1]]
end

type QDAClassifier

    cats::Array

    priors::Vector
    centers::Matrix

    precisions::Array
    const_term::Vector
end

function StatsBase.fit(::Type{QDAClassifier}, X::Matrix, y::Vector)
    n = size(y)[1]
    cats = unique(y)
    n_cats = length(cats)
    priors = [1.0*count(k -> k == i, y)/n for i in cats]
    centers = zeros(size(X)[2], n_cats); 
    precisions = Matrix[]
    for (i, x) in enumerate(cats)
        centers[:, i] = mean(X[y .== x, :], 1)
        push!(precisions, cov(X[y .== x, :])^-1)
    end
    const_term = [-.5*log(det(precisions[i]^-1)) + log(priors[i]) for i in 1:n_cats]

    return QDAClassifier(cats, priors, centers, precisions, const_term)
end

function StatsBase.predict(mm::QDAClassifier, newX::Matrix)
    return [mm.cats[indmax([(-.5*(newX[n, :] - mm.centers[:, i]')*mm.precisions[i]*(newX[n, :]' - mm.centers[:, i]) + mm.const_term[i])[1] for i in 1:length(mm.priors)])] for n in 1:size(newX)[1]]
end

type RRLDAClassifier
    lda::LDAClassifier
    V::Matrix
end

function StatsBase.fit(::Type{RRLDAClassifier}, X::Matrix, y::Vector, k::Integer)
    n = size(y)[1]
    p = size(X)[2]
    n_cats = maximum(y)
    priors = [1.0*count(l -> l == i, y)/n for i in 1:n_cats]
    centers = zeros(n_cats, p); 
    for i in 1:n_cats
        centers[i, :] = mean(X[y .== i, :], 1)
    end
    #W = cov(X)
    W = sum([(X[i, :] - centers[y[i], :])'*(X[i, :] - centers[y[i], :]) for i in 1:size(X)[1]])/(n - n_cats)
    W_Minus_One_Half = W^-.5
    M_star = centers*W_Minus_One_Half
    B_star = cov(M_star)
    V = (W_Minus_One_Half*svd(B_star)[1])[:, 1:k]

    Z = X*V
    return RRLDAClassifier(fit(LDAClassifier, Z, y), V)
end

function StatsBase.predict(mm::RRLDAClassifier, newX::Matrix)
    return predict(mm.lda, newX*mm.V)
end

type LogLinearClassifier
    beta::Vector
end

function StatsBase.fit(::Type{LogLinearClassifier}, X::Matrix, y::Vector)
    n = size(y)[1]
    p = size(X)[2]
    n_cats = maximum(y)

    padded_x = [ones(n) X]
    beta = zeros(p + 1)
    while true
        P = [e^((padded_x[i,:]*beta)[1])/(1 + e^((padded_x[i,:]*beta)[1])) for i in 1:n]
        W = diagm(P .* (1 - P))
        beta_new = (padded_x'*W*padded_x)^-1 * padded_x' * W * (padded_x*beta + W^-1*(y - P))
        if (((beta_new - beta)'*(beta_new - beta))[1] < .00001)
            beta = beta_new
            break
        end
        beta = beta_new
    end
    return LogLinearClassifier(beta)
end

function StatsBase.predict(mm::LogLinearClassifier, newX::Matrix)
    n = size(newX)[1]
    padded_x = [ones(n) newX]
    return [round(e^((padded_x[i,:]*mm.beta)[1])/(1 + e^((padded_x[i,:]*mm.beta)[1]))) for i in 1:n]
end

end
