module Ch4

using Classification
using DataFrames, DataFramesUtil
using Gadfly

function load_data(fileName::String, separator::Char, header::Bool, response::Symbol, standardize=false, ignoreCols=Symbol[], classCols=Symbol[])
    df = readtable(fileName, separator=separator, header=header)
    X = df[filter(x -> !(x in ignoreCols) && x != response, names(df))]
    for col in classCols
        X[col] = pool(X[col]).refs - 1
    end
    if (standardize)
        X = DataFramesUtil.standardize!(X)
    end
    y = df[response]

    return convert(Matrix{Float64}, X), convert(Vector{Int64}, y)
end

function figure_4_4()
    trainX, trainY = load_data("../data/vowels.data", ',', true, :y, true, [:row_names])
    rrlda = fit(Classification.RRLDAClassifier, trainX, trainY, 10)
    Z = trainX*rrlda.V
    Gadfly.plot(x=Z[:,1], y=Z[:,7], color=trainY, Geom.point)
end

function figure_4_10()
    trainX, trainY = load_data("../data/vowels.data", ',', true, :y, false, [:row_names])
    testX, testY = load_data("../data/vowels.test", ',', true, :y, false, [:row_names])

    trainErr = Real[]; testErr = Real[]
    for i in 1:10
        rrlda = fit(Classification.RRLDAClassifier, trainX, trainY, i)
        push!(trainErr, (1 - count(x -> x == 0, trainY - predict(rrlda, trainX))/size(trainY)[1]))
        push!(testErr, (1 - count(x -> x == 0, testY - predict(rrlda, testX))/size(testY)[1]))
    end
    Gadfly.plot(layer(x=1:10, y=trainErr, Geom.point, Geom.line, Theme(default_color=color("blue"))),
                layer(x=1:10, y=testErr, Geom.point, Geom.line, Theme(default_color=color("orange"))),
                Guide.xlabel("Dimension"), Guide.ylabel("Missclassification Rate"), Guide.title("RRLDA on Vowel Data"))
end

function table_4_2()
    X, y = Ch4.load_data("../data/SAheart.data", ',', true, :chd, false, [:row_names, :adiposity, :typea], [:famhist])
    llc = fit(Classification.LogLinearClassifier, X, y)
    print(DataFrame(Coefficient=llc.beta))
end

end
