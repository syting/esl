module Splines

# Generates a cubic basis spline for X. If knots are not specified by the user they are
# set to X
function bs(X, knots = [], intercept = false)
    if isempty(knots)
        knots = X
    end
    n = length(X)
    M = length(knots) + (intercept ? 4 : 3)
    N = zeros(n, M)
    j = 1
    if intercept
        N[:, j] = ones(n); j += 1
    end
    N[:, j] = X; j += 1
    N[:, j] = X.^2; j += 1
    N[:, j] = X.^3; j += 1
    for knot in knots
        N[:, j] = [max(0, (x - knot)^3) for x in X]; j += 1
    end
    return N
end

# Generates n-quantiles of the data in X
function quantiles(X, n)
    sorted_X = sort(X)
    return [sorted_X[max(1, div(i*size(X)[1], n))] for i in 0:n]
end

function ns_funcs(knots, intercept = false, omega = false)
    funcs = Function[]
    if intercept
        push!(funcs, omega ? x -> 0.0 : x -> 1.0)
    end
    push!(funcs, omega ? x -> 0.0 : x -> x)
    last_knot = knots[end]
    function d(x, z)
        if !omega
            return (max(0, (x - z)^3) - max(0, (x - last_knot)^3))/(last_knot - z)
        else
            return min(6, max(0, 6*(x - z)/(last_knot - z)))
        end
    end
    for i in 1:(length(knots) - 2)
        push!(funcs, x -> d(x, knots[i]) - d(x, knots[end-1]))
    end
    return funcs
end

# Generates a natural cubic spline for X. If knots are not given then they are set at the unique
# points in X
function ns(X, knots = [], intercept = false)
    if isempty(knots)
        knots = unique(sort(X))
    end
    n = length(X)
    basis_funcs = ns_funcs(knots, intercept, false)
    M = length(basis_funcs)
    N = zeros(n, M)
    for j in 1:length(basis_funcs)
        N[:, j] = [basis_funcs[j](x) for x in X]
    end
    return N
end

function calc_omega(X)
    n = length(X)
    omega = zeros(n, n)
    Npp1(x, i, j) = 36/((X[end] - X[i])*(X[end] - X[j])) * (x^3/3 - .5*(X[i]+X[j])*x^2 + X[i]*X[j]*x)
    Npp2(x, i, j) = 36*(X[end-1] - X[i])*(X[end-1] - X[j])/((X[end]-X[end-1])^2*(X[end]-X[i])*(X[end]-X[j]))*(x^3/3 -X[end]*x^2 + X[end]^2*x)
    Npp(i, j) = Npp1(X[end-1], i, j) - Npp1(X[max(i,j)], i, j) + Npp2(X[end], i, j) - Npp2(X[end-1], i, j)

    for i in 3:n
        for j in 3:n
            omega[i, j] = Npp(i-2, j-2)
        end
    end
    return omega
end

# Generates a smoothing spline for design X and data y using the given lambda. If M is not specified
# knots are placed at all points in X, otherwise M knots are placed at the appropriate quantiles of X.
# The fitted N and theta values are returned
function smooth_spline(X, y, lambda, M=0)
    knots = unique(sort(X))
    if 0 < M
        knots = quantiles(unique(X), M)
    end
    N = ns(X, knots, true)
    omega = calc_omega(knots)
    return N, (N'*N + lambda*omega)^-1*N'*y
end

end
