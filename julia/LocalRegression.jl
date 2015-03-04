module LocalRegression

function NWKernel(x0, X, y, K)
    sum([K(x0, x)*y for (x, y) in zip(X, y)])/sum([K(x0, x) for x in X])
end

function k_nearest(x0, x, X, k)
    return (norm(x - x0) <= sort(norm(X - x0))[k]) ? 1.0 : 0.0
end

function Epanechnikov(x0, x, l)
    t = norm(x - x0)/l
    return (t <= 1.0) ? .75*(1 - t^2) : 0.0
end

function tri_cube_kernel(x0, x, l)
    t = norm(x - x0)/l
    return (t <= 1.0) ? (1 - t^3)^3 : 0.0
end

function lwpr_l(x0, X, K, p)
    W = diagm(convert(Array{Float64}, [K(x0, x) for x in X]))
    b = zeros(p + 1); b[1] = 1
    B = ones(length(X), p + 1)
    for (i,x) in enumerate(X)
        for j in 2:(p+1)
            B[i,j] = (x - x0)^(j-1)
        end
    end
    return b'*(B'*W*B)^-1*B'*W
end

function lwpr(x0, X, y, K, p)
    l = lwpr_l(x0, X, K, p)
    return (l*y)[1]
end

function llda(x0, X, y, K)
    n = size(y)[1]
    n_cats = length(unique(y))
    weights = convert(Array{Float64}, [K(x0, X[i, :]) for i in 1:n])
    W = diagm(weights)
    total_class_weights = [sum(weights[y .== i]) for i in 0:(n_cats-1)]
    centers = zeros(size(X)[2], n_cats)
    for i in 0:(n_cats - 1)
        centers[:, i+1] = sum(W[y .== i, y .== i]'*X[y .== i,:]/total_class_weights[i+1], 1)
    end
    covariance = sum([weights[i]*(X[i, :]' - centers[:, y[i]+1])*(X[i, :]' - centers[:, y[i]+1])' for i in 1:size(X)[1]])/sum(weights)
    precision = covariance^-1
    const_term = [-.5*(centers[:, i]'*precision*centers[:, i])[1] + log(weights[i]) for i in 1:n_cats]

    discr_vals = x0*precision*centers .+ const_term'
    return indmax(discr_vals) - 1
end

end
