module Ch4

using Classification
using DataFrames, DataFramesUtil
using ESLii
using Gadfly

function figure_4_4_and_4_8(i, j)
    trainX, trainY = ESLii.read_vowel_data()
    rrlda = fit(Classification.RRLDAClassifier, trainX, trainY, 10)
    Z = trainX*rrlda.V
    Gadfly.plot(x=Z[:, i], y=Z[:, j], color=trainY, Geom.point)
end

function figure_4_10()
    trainX, trainY = ESLii.load_data("../data/vowels.data", ',', true, :y, false, [:row_names])
    testX, testY = ESLii.load_data("../data/vowels.test", ',', true, :y, false, [:row_names])

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
    X, y = Ch4.ESLii.load_data("../data/SAheart.data", ',', true, :chd, false, [:row_names, :adiposity, :typea], [:famhist])
    llc = fit(Classification.LogLinearClassifier, X, y)
    print(DataFrame(Coefficient=llc.beta))
end

end
