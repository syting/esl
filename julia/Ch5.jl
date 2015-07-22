module Ch5

using Classification
using DataFrames, DataFramesUtil
using ESLii
using Gadfly
using Splines

function pv(X)
    cv = (X'*X)^-1
    return [(X[i,:]*cv*X[i,:]')[1] for i in 1:size(X)[1]]
end

function figure_5_3()
    X = rand(50)
    L_H = [ones(X) X]
    l_pv = pv(L_H)
    C_H = [ones(X) X X.^2 X.^3]
    c_pv = pv(C_H)
    CS_H = Splines.bs(X, [.33, .66], true)
    cs_pv = pv(CS_H)
    NCS_H = Splines.ns(X, [.1, .26, .42, .58, .74, .9], true)
    ncs_pv = pv(NCS_H)

    Gadfly.plot(layer(x=X, y=l_pv, Theme(default_color=color("orange")), Geom.point, Geom.line),
                layer(x=X, y=c_pv, Theme(default_color=color("red")), Geom.point, Geom.line),
                layer(x=X, y=ncs_pv, Theme(default_color=color("blue")), Geom.point, Geom.line),
                layer(x=X, y=cs_pv, Theme(default_color=color("green")), Geom.point, Geom.line),
                Guide.xlabel("X"), Guide.ylabel("Pointwise Variance"))
end

function quartiles(X)
    sorted_x = sort(X)
    return [sorted_x[max(1, div(i*size(X)[1], 4))] for i in 0:4]
end

function quantiles(X, n)
    sorted_x = sort(X)
    return [sorted_x[max(1, div(i*size(X)[1], n))] for i in 0:n]
end

function figure_5_4()
    X, y = ESLii.load_data("../data/SAheart.data", ',', true, :chd, false, [:row_names, :adiposity, :typea], [:famhist])
    N1 = Splines.ns(X[:,1], quartiles(X[:,1]), false)
    N2 = Splines.ns(X[:,2], quartiles(X[:,2]), false)
    N3 = Splines.ns(X[:,3], quartiles(X[:,3]), false)
    N4 = X[:,4]
    N5 = Splines.ns(X[:,5], quartiles(X[:,5]), false)
    N7 = Splines.ns(X[:,7], quartiles(X[:,7]), false)
    N = [N1 N2 N3 N4 N5 N7]
    llc = fit(Classification.LogLinearClassifier, N, y)

    for j in 1:size(N)[2]
        N[:, j] = N[:, j] - mean(N[:, j])
    end
    Gadfly.plot(x=X[:, 1], y=N[:, 1:4]*llc.beta[2:5], Geom.line)
    Gadfly.plot(x=X[:, 2], y=N[:, 5:8]*llc.beta[6:9], Geom.line)
    Gadfly.plot(x=X[:, 3], y=N[:, 9:12]*llc.beta[10:13], Geom.line)
    Gadfly.plot(x=X[:, 4], y=N[:, 13]*llc.beta[14], Geom.line)
    Gadfly.plot(x=X[:, 5], y=N[:, 14:17]*llc.beta[15:18], Geom.line)
    Gadfly.plot(x=X[:, 7], y=N[:, 18:end]*llc.beta[19:end], Geom.line)
end

function figure_5_5()
    phoneme = readtable("../data/phoneme.data", separator=',', header=true)
    aa = phoneme[phoneme[:g] .== "aa", :]
    aa_train = aa[convert(Array{Bool}, map(x -> contains(x, "train"), aa[:speaker])), :]
    aa_test = aa[convert(Array{Bool}, map(x -> contains(x, "test"), aa[:speaker])), :]
    ao = phoneme[phoneme[:g] .== "ao", :]
    ao_train = ao[convert(Array{Bool}, map(x -> contains(x, "train"), ao[:speaker])), :]
    ao_test = ao[convert(Array{Bool}, map(x -> contains(x, "test"), ao[:speaker])), :]

    # Print some examples of the data
    for i in 1:15
        Gadfly.plot(x=1:256, y=aa_train[i, 1:256])
        Gadfly.plot(x=1:256, y=ao_train[i, 1:256])
    end
end

end
