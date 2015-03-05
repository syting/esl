module LinearRegression

using DataFrames, GLM, LBP, NumericExtensions

"""Performs leaps and bounds regression on the given DataFrame returning a dictionary
keyed by subset size with values containing the optimal index set at that size
"""
function lbp(df::DataFrame)
    # Eventually I should add the option to specify which indices to include as predictors,
    # whether to include an intercept, and a specific index for the response
    X = convert(Matrix{Float64}, df[:,1:end-1])
    Y = convert(Vector{Float64}, df[end])
    adjX = [ones(size(X,1)) X]
    return LBP.lbpRegress(adjX, Y)
end


"""Performs all subsets regression on the given DataFrame D with column indexed
y containing the response
"""
function all_subsets_regression(D::DataFrame, y::Symbol)
    N = size(D)[1]
    p = size(D)[2]

    # Perform k=0 regression
    results = [(convert(Array{Symbol}, []), sumsq(residuals(lm(Formula(y, Expr(:call, :+, 1)), D))))]

    # Perform regressions over all subsets of indices
    for k = 1:(size(D,2) - 1)
        for ss in combinations(filter(x -> x != y, names(D)), k)
            push!(results, (ss, sumsq(residuals(lm(Formula(y, Expr(:call, :+, ss)), D)))))
        end
    end
    return results
end

best_k_subset(k, df, y) = reduce((x,y) -> x[2] < y[2] ? x : y, filter(x -> size(x[1], 1) == k, all_subsets_regression(df, y)))
best_k_subset(k, results) = reduce((x,y) -> x[2] < y[2] ? x : y, filter(x -> x[1] == k, results))

""" Performs a Least-Angle-Regression given predictors X and response y
"""
function lar(X, y)
    # Standardize the inputs
    pred = standardize(X)
    resp = (y - mean(y))/std(y)

    # Initialize residual, active and inactive sets
    r = resp
    first_pred = indmax(cov(pred,resp))
    active_set = [first_pred]
    inactive_set = [j for j in 1:size(pred)[2]]
    splice!(inactive_set, first_pred)
    beta = [0.0]

    # Repeatedly add variables to active set when their covariance with the residual matches that of
    # the active variables
    while 0 < length(inactive_set)
        delta = (pred[:,active_set]'*pred[:,active_set])^-1*pred[:,active_set]'*r # Current direction
        rh = pred[:,active_set]*delta # Fitted residuals
        l = cov(pred[:, active_set[1]], r, corrected=false) # Current covariance

        # Find the distance to travel along delta until an inactive variable has equal correlation with the 
        # residual with the active variables
        alphas = [min((l - cov(pred[:,j], r, corrected=false))/(l - cov(pred[:,j], rh, corrected=false)),
                      (l + cov(pred[:,j], r, corrected=false))/(l + cov(pred[:,j], rh, corrected=false))) for j in inactive_set]
        alpha_ind = indmin(alphas)
        alpha = alphas[alpha_ind]

        # Update beta, our active/inactive sets, and the residual
        beta = [beta; 0.0] + alpha*[delta; 0.0]
        push!(active_set, inactive_set[alpha_ind])
        splice!(inactive_set, alpha_ind)
        r = resp - pred[:, active_set]*beta
    end
    return active_set
end

""" Standardizes D to have mean 0 and std 1
"""
function standardize{T<:Real}(D::Matrix{T})
    standardized_data = zeros(size(D))
    for j in 1:size(D)[2]
        standardized_data[:,j] = (D[:,j] - mean(D[:,j]))/std(D[:,j])
    end
    return standardized_data
end

""" Performs principal component regression using M components over response y with 
predictors X
"""
function pcr(X, y, M)
    # Standardize the inputs
    pred = standardize(X)
    y_mean = mean(y)
    resp = (y - mean(y))

    U, D, V = svd(pred)
    Z = pred*V[:,1:M]
    beta = zeros(size(pred)[2])
    for i in 1:M
        beta += (Z[:,i]'*resp)[1]/(Z[:,i]'*Z[:,i])[1]*V[:,i]
    end
    return [y_mean; beta]
end

function pls(X, y, M)
    # Standardize the inputs
    pred = standardize(X)
    y_mean = mean(y)
    resp = y

    cur_pred = pred
    cur_resp = ones(size(y)[1])*y_mean
    for i in 1:M
        z_i = cur_pred*cur_pred'*resp
        cur_resp += (z_i'*resp)[1]/(z_i'*z_i)[1]*z_i
        cur_pred -= z_i*z_i'cur_pred/(z_i'*z_i)[1]
    end
    return [y_mean; (pred'*pred)^-1*pred'*cur_resp]
end

end
