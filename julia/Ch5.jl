module Ch5

using Classification
using DataFrames, DataFramesUtil
using ESLii
using Gadfly
using Splines

# Calculates the pointwise-variance of the design matrix X
function pv(X)
    cv = (X'*X)^-1
    return [(X[i,:]*cv*X[i,:]')[1] for i in 1:size(X)[1]]
end

# Reproduces figure 5.3 in ESLii displaying the pointwise-variance curves for four different spline
# fits of X where X consists of 50 randomly chosen points in U[0,1]
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

# Reproduces Figure 5.4 in ESLii displaying fitted natural-spline function for each term
# in the final model of a loglinear fit to the South African heart disease data
function figure_5_4()
    X, y = ESLii.read_sa_heart_data()
    N1 = Splines.ns(X[:,1], Splines.quantiles(X[:,1], 4), false)
    N2 = Splines.ns(X[:,2], Splines.quantiles(X[:,2], 4), false)
    N3 = Splines.ns(X[:,3], Splines.quantiles(X[:,3], 4), false)
    N4 = X[:,4]
    N5 = Splines.ns(X[:,5], Splines.quantiles(X[:,5], 4), false)
    N7 = Splines.ns(X[:,7], Splines.quantiles(X[:,7], 4), false)
    N = [N1 N2 N3 N4 N5 N7]
    llc = fit(Classification.LogLinearClassifier, N, y)

    for j in 1:size(N)[2]
        N[:, j] = N[:, j] - mean(N[:, j])
    end
    p1 = Gadfly.plot(x=X[:, 1], y=N[:, 1:4]*llc.beta[2:5], Geom.line, Guide.xlabel("sbp"), Guide.ylabel("f^(sbp)"))
    p2 = Gadfly.plot(x=X[:, 2], y=N[:, 5:8]*llc.beta[6:9], Geom.line, Guide.xlabel("tobacco"), Guide.ylabel("f^(tobacco)"))
    p3 = Gadfly.plot(x=X[:, 3], y=N[:, 9:12]*llc.beta[10:13], Geom.line, Guide.xlabel("ldl"), Guide.ylabel("f^(ldl)"))
    p4 = Gadfly.plot(x=X[:, 4], y=N[:, 13]*llc.beta[14], Geom.point, Guide.xlabel("famhist"), Guide.ylabel("f^(famhist)"))
    p5 = Gadfly.plot(x=X[:, 5], y=N[:, 14:17]*llc.beta[15:18], Geom.line, Guide.xlabel("obesity"), Guide.ylabel("f^(obesity)"))
    p6 = Gadfly.plot(x=X[:, 7], y=N[:, 18:end]*llc.beta[19:end], Geom.line, Guide.xlabel("age"), Guide.ylabel("f^(age)"))
    Gadfly.draw(SVG("Figure_5_4.svg", 4inch, 6inch), hstack(vstack(p1, p3, p5), vstack(p2, p4, p6)))
end

# Reproduces Figure 5.5 in ESLii displaying the results of fitting a loglinear classifier to phoneme
# data distinguishing the "aa" sound from the "ao" sound using the raw data versus data regularized
# with a natural cubic spline fit.
function figure_5_5()
    phoneme = readtable("../data/phoneme.data", separator=',', header=true)
    phoneme = phoneme[filter(x -> !(x in [:row_names]), names(phoneme))]
    aa = phoneme[phoneme[:g] .== "aa", :]
    aa_train = aa[convert(Array{Bool}, map(x -> contains(x, "train"), aa[:speaker])), :]
    aa_test = aa[convert(Array{Bool}, map(x -> contains(x, "test"), aa[:speaker])), :]
    ao = phoneme[phoneme[:g] .== "ao", :]
    ao_train = ao[convert(Array{Bool}, map(x -> contains(x, "train"), ao[:speaker])), :]
    ao_test = ao[convert(Array{Bool}, map(x -> contains(x, "test"), ao[:speaker])), :]

    # Print some examples of the data
    p1 = Gadfly.plot(x=1:256, y=round(vec(convert(Array, aa_train[1, 1:256])), 2), Geom.line, Theme(default_color=color("green")), Guide.xlabel("Frequency"), Guide.ylabel("Log-periodogram"), Guide.title("Phoneme Examples"))
    push!(p1.layers, layer(x=1:256, y=round(vec(convert(Array, ao_train[1, 1:256])), 2), Geom.line, Theme(default_color=color("red")))[1])
    for i in 2:5
        push!(p1.layers, layer(x=1:256, y=round(vec(convert(Array, aa_train[i, 1:256])), 2), Geom.line, Theme(default_color=color("green")))[1])
        push!(p1.layers, layer(x=1:256, y=round(vec(convert(Array, ao_train[i, 1:256])), 2), Geom.line, Theme(default_color=color("red")))[1])
    end

    # Separate out train and test data/labels
    train_X = convert(Matrix, [aa_train[:, 1:256], ao_train[:, 1:256]])
    train_y = [i < size(aa_train)[1] ? 0 : 1 for i in 1:size(train_X)[1]]
    test_X = convert(Matrix, [aa_test[:, 1:256], ao_test[:, 1:256]])
    test_y = [i < size(aa_test)[1] ? 0 : 1 for i in 1:size(test_X)[1]]

    # Train raw classifier
    raw_llc = fit(Classification.LogLinearClassifier, train_X, train_y)
    raw_train_err = 1 - count(x -> x == 0, train_y - predict(raw_llc, train_X))/size(train_y)[1]
    raw_test_err = 1 - count(x -> x == 0, test_y - predict(raw_llc, test_X))/size(test_y)[1]
    print("Raw | Training Error: $(raw_train_err)  Test Error: $(raw_test_err)\n")

    # Train regularized classifier
    N = Splines.ns(1:256, [1, 21, 42, 64, 85, 106, 128, 149, 170, 192, 213, 234, 256], true)
    reg_llc = fit(Classification.LogLinearClassifier, train_X*N, train_y)
    reg_train_err = 1 - count(x -> x == 0, train_y - predict(reg_llc, train_X*N))/size(train_y)[1]
    reg_test_err = 1 - count(x -> x == 0, test_y - predict(reg_llc, test_X*N))/size(test_y)[1]
    print("Reg | Training Error: $(reg_train_err)  Test Error: $(reg_test_err)")

    p2 = Gadfly.plot(layer(x=1:256, y=raw_llc.beta[2:end], Geom.line, Theme(default_color=color("grey"))),
    layer(x=1:256, y=N*reg_llc.beta[2:end], Geom.line, Theme(default_color=color("red"))),
    Guide.xlabel("Frequency"), Guide.ylabel("Logistic Regression Coefficients"),
    Guide.title("Phoneme Classification: Raw and Restricted Logistic Regression"))

    Gadfly.draw(SVG("Figure_5_5.svg", 6inch, 8inch), vstack(p1, p2))
end

# Reproduces Figure 5.6 in ESLii showing a smoothing spline fit to the relative change in spinal bmd
# versus for males and females. Because of the point size in the Gadfly package the raw data is excluded
# and only the fitted curves are shown
function figure_5_6()
    bone = ESLii.read_bone_data()
    male_bone = sort(bone[bone[:gender] .== "male", :], cols=:age)
    female_bone = sort(bone[bone[:gender] .== "female", :], cols=:age)

    #male_N = Splines.ns(male_bone[:age], Splines.quantiles(male_bone[:age], 12), true)
    #male_omega = Splines.calc_omega(Splines.quantiles(male_bone[:age], 12))
    #male_theta = (male_N'*male_N + .00022*male_omega)^-1*male_N'*male_bone[:spnbmd]
    #female_N = Splines.ns(female_bone[:age], Splines.quantiles(female_bone[:age], 12), true)
    #female_omega = Splines.calc_omega(Splines.quantiles(female_bone[:age], 12))
    #female_theta = (female_N'*female_N + .00022*female_omega)^-1*female_N'*female_bone[:spnbmd]
    male_N, male_theta = Splines.smooth_spline(male_bone[:age], male_bone[:spnbmd], .00022, 10)
    female_N, female_theta = Splines.smooth_spline(female_bone[:age], female_bone[:spnbmd], .00022, 10)

    #Gadfly.plot(layer(x=male_bone[:age], y=male_bone[:spnbmd], Geom.point, Theme(default_color=color("blue"))),
    Gadfly.plot(layer(x=male_bone[:age], y=male_N*male_theta, Geom.line, Theme(default_color=color("blue"))),
    #layer(x=female_bone[:age], y=female_bone[:spnbmd], Geom.point, Theme(default_color=color("red"))),
    layer(x=female_bone[:age], y=female_N*female_theta, Geom.line, Theme(default_color=color("red"))),
    Guide.xlabel("Age"), Guide.ylabel("Relative Change in Spinal BMD"))
end

function figure_5_9()
    X = rand(100)
    f_X = sin(12(X + 0.2))./(X + 0.2)
    y = f_X + randn(100)

    Gadfly.plot(layer(x=X, y=y, Geom.point),
                layer(x=X, y=f_X, Geom.line, Theme(default_color=color("purple"))))
end

end
