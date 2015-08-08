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

function calc_omega(X, intercept = false)
    n = length(X)
    basis_funcs = ns_funcs(X, intercept, true)
    omega = zeros(n, n)
    for i in 1:n
        for j in 1:n
            omega[i, j] = quadgk(x -> basis_funcs[i](x)*basis_funcs[j](x), X[1], X[end])[1]
        end
    end
    return omega
end

function calc_omega2(X)
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

function b_spline_funcs(X, M)
    Xm = unique(sort(X))
    K = length(Xm) - 2
    Z = Xm[1] * ones(K + 2*M); Z[M:M+K+1] = Xm; Z[M+K+1:end] = Xm[end]
    basis_funcs = Array{Function}[]
    push!(basis_funcs, [x -> (Z[i] <= x < Z[i+1]) ? 1.0 : 0.0 for i in 1:(K + 2*M - 1)])
    function B(x, i, m)
        if Z[i] == Z[i + m]
            return 0.0
        elseif Z[i] == Z[i + m - 1]
            return basis_funcs[m-1][i+1](x)*(Z[i+m] - x)/(Z[i+m] - Z[i+1])
        elseif Z[i + 1] == Z[i + m]
            return basis_funcs[m-1][i](x)*(x - Z[i])/(Z[i+m-1] - Z[i])
        end
        return basis_funcs[m-1][i](x)*(x - Z[i])/(Z[i+m-1] - Z[i]) + basis_funcs[m-1][i+1](x)*(Z[i+m] - x)/(Z[i+m] - Z[i+1])
    end
    for m in 2:M
        new_funcs = Function[]
        for i in 1:(K + 2*M - m)
            push!(new_funcs, x -> B(x, i, m))
        end
        push!(basis_funcs, new_funcs)
    end
    return basis_funcs
end

function b_splines(X, M)
    basis_funcs = b_spline_funcs(X, M)[end]
    n = length(X)
    K = length(basis_funcs)
    N = zeros(n, K)
    for i in 1:n
        for j in 1:K
            N[i,j] = basis_funcs[j](X[i])
        end
    end
    return N
end

end
