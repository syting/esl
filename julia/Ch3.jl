module Ch3

using DataFrames, DataFramesUtil
using Gadfly
using GLM
using LinearRegression
using MLBase
using MultivariateStats
using Roots

function load_data()
    df = readtable("../data/prostate.data", separator=' ', header=false, skipstart=1)
    df = df[:,2:end]
    rename!(df, names(df), [:lcavol, :lweight, :age, :lbph, :sci, :lcp, :gleason, :ppg45, :lpsa, :train])
    return df
end

function cleaned_data()
    # Load and standardize the data, then split into train and test examples
    df = load_data()
    standardized_data = copy(df)
    standardized_data[1:8] = DataFramesUtil.standardize!(df[1:8])
    train_data = standardized_data[standardized_data[:train] .== true,:][1:end - 1]
    test_data = standardized_data[standardized_data[:train] .== false,:][1:end - 1]

    return train_data, test_data
end

function load_spam_data()
    df = readtable("../data/spam.data", separator=' ', header=false, skipstart=0)
    df = df[:,2:end]
    return df
end

function cleaned_spam_data()
    # Load and standardize the data, then split into train and test examples
    df = load_spam_data()
    standardized_data = copy(df)
    standardized_data[1:end - 1] = DataFramesUtil.standardize!(df[1:end - 1])

    return standardized_data
end

function table_3_1()
    df = load_data()
    df = df[1:9]
    cor(df)
end

function table_3_2()
    train_data, test_data = cleaned_data()
    model = lm(lpsa~lcavol+lweight+age+lbph+sci+lcp+gleason+ppg45, train_data)
end

function figure_3_5()
    train_data, test_data = cleaned_data()
    all_regressions = map(x -> (length(x[1]), x[2]), LinearRegression.all_subsets_regression(train_data, :lpsa))
    Gadfly.plot(layer(x=map(x -> x[1], all_regressions), y=map(x -> x[2], all_regressions), Geom.point),
                layer(x=0:(size(train_data)[2] - 1), y=map(k -> LinearRegression.best_k_subset(k, all_regressions)[2], 0:(size(train_data)[2] - 1)), Geom.line, Theme(default_color=color("red"))),
                Guide.xlabel("Num Predictors"), Guide.ylabel("Residual Sum-of-Squares"))
end

function lambda_ridge(X, multiple=1)
    dj = svd(X)[2]

    lambdas = [0.0]
    for k in length(dj)-1:-multiple:0
        f(x) = sum(map(i -> dj[i]^2/(dj[i]^2 + x), 1:length(dj))) - k
        fp(x) = sum(map(i -> -dj[i]^2/(dj[i]^2 + x)^2, 1:length(dj)))
        r = newton(f, fp, lambdas[end])
        push!(lambdas, r)
    end

    return reverse(lambdas)
end

function figure_3_7_ridge()
    train_data, test_data = cleaned_data()
    mf = ModelFrame(Formula(names(train_data)[end], Expr(:call, :+, names(train_data)[1:end-1])), train_data)
    X = ModelMatrix(mf).m[:,2:end]
    N = size(X)[1]
    y = model_response(mf)

    cv_results = (Number, Array)[]
    lambdas = lambda_ridge(X)
    for lambda in lambdas
        lcl_ridge(indices) = ridge(X[indices,:], y[indices], lambda)
        ridge_score(A, indices) = sum((y[indices] - A[end] - X[indices,:]*A[1:end-1]).^2)/length(indices)
        push!(cv_results, (lambda, cross_validate(lcl_ridge, ridge_score, N, Kfold(N, 10))))
    end

    means = map(x -> mean(x[2]), cv_results)
    stds = map(x -> std(x[2]), cv_results)
    Gadfly.plot(layer(x=0:(length(lambdas)-1), y=means, Geom.point, Geom.line, Theme(default_color=color("orange"))),
                layer(x=0:(length(lambdas)-1), ymin=(means - .5*stds), ymax=(means + .5*stds), Geom.errorbar),
                Guide.xlabel("Degrees of Freedom"), Guide.ylabel("CV Error"), Guide.title("Ridge Regression"))
end

function figure_3_7_pcr()
    train_data, test_data = cleaned_data()
    mf = ModelFrame(Formula(names(train_data)[end], Expr(:call, :+, names(train_data)[1:end-1])), train_data)
    X = ModelMatrix(mf).m[:,2:end]
    N = size(X)[1]
    y = model_response(mf)

    cv_results = (Number, Array)[]
    for dof in 0:size(X)[2]
        lcl_pcr(indices) = LinearRegression.pcr(X[indices,:], y[indices], dof)
        pcr_score(A, indices) = sum((y[indices] - A[1] - X[indices,:]*A[2:end]).^2)/length(indices)
        push!(cv_results, (dof, cross_validate(lcl_pcr, pcr_score, N, Kfold(N, 10))))
    end

    means = map(x -> mean(x[2]), cv_results)
    stds = map(x -> std(x[2]), cv_results)
    Gadfly.plot(layer(x=0:size(X)[2], y=means, Geom.point, Geom.line, Theme(default_color=color("orange"))),
                layer(x=0:size(X)[2], ymin=(means - .5*stds), ymax=(means + .5*stds), Geom.errorbar),
                Guide.xlabel("Degrees of Freedom"), Guide.ylabel("CV Error"), Guide.title("PCR"))
end

function figure_3_7_pls()
    train_data, test_data = cleaned_data()
    mf = ModelFrame(Formula(names(train_data)[end], Expr(:call, :+, names(train_data)[1:end-1])), train_data)
    X = ModelMatrix(mf).m[:,2:end]
    N = size(X)[1]
    y = model_response(mf)

    cv_results = (Number, Array)[]
    for dof in 0:size(X)[2]
        lcl_pls(indices) = LinearRegression.pls(X[indices,:], y[indices], dof)
        pls_score(A, indices) = sum((y[indices] - A[1] - X[indices,:]*A[2:end]).^2)/length(indices)
        push!(cv_results, (dof, cross_validate(lcl_pls, pls_score, N, Kfold(N, 10))))
    end

    means = map(x -> mean(x[2]), cv_results)
    stds = map(x -> std(x[2]), cv_results)
    Gadfly.plot(layer(x=0:size(X)[2], y=means, Geom.point, Geom.line, Theme(default_color=color("orange"))),
                layer(x=0:size(X)[2], ymin=(means - .5*stds), ymax=(means + .5*stds), Geom.errorbar),
                Guide.xlabel("Degrees of Freedom"), Guide.ylabel("CV Error"), Guide.title("PLS"))
end


end
