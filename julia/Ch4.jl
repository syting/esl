module Ch4

using Classification
using DataFrames, DataFramesUtil
using ESLii
using Gadfly

function table_4_1()
    trainX, trainY = ESLii.read_vowel_data()
    testX, testY = ESLii.read_vowel_data(false)

    lr = fit(Classification.LRClassifier, trainX, trainY)
    lrTrainErr = (1 - count(x -> x == 0, trainY - predict(lr, trainX))/size(trainY)[1])
    lrTestErr = (1 - count(x -> x == 0, testY - predict(lr, testX))/size(testY)[1])
    print("Linear Regression: $(lrTrainErr) $(lrTestErr)\n")

    lda = fit(Classification.LDAClassifier, trainX, trainY)
    ldaTrainErr = (1 - count(x -> x == 0, trainY - predict(lda, trainX))/size(trainY)[1])
    ldaTestErr = (1 - count(x -> x == 0, testY - predict(lda, testX))/size(testY)[1])
    print("Linear Discriminant Analysis: $(ldaTrainErr) $(ldaTestErr)\n")

    qda = fit(Classification.QDAClassifier, trainX, trainY)
    qdaTrainErr = (1 - count(x -> x == 0, trainY - predict(qda, trainX))/size(trainY)[1])
    qdaTestErr = (1 - count(x -> x == 0, testY - predict(qda, testX))/size(testY)[1])
    print("Quadratic Discriminant Analysis: $(qdaTrainErr) $(qdaTestErr)\n")
end

function figure_4_4_and_4_8(i, j)
    trainX, trainY = ESLii.read_vowel_data()
    rrlda = fit(Classification.RRLDAClassifier, trainX, trainY, 10)
    Z = (rrlda.V'*trainX')'
    Gadfly.plot(x=Z[:, i], y=Z[:, j], color=trainY, Geom.point)
end

function figure_4_10()
    trainX, trainY = ESLii.read_vowel_data()
    testX, testY = ESLii.read_vowel_data(false)

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
