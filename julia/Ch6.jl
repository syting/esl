module Ch6

using Classification
using DataFrames, DataFramesUtil
using Distributions
using ESLii
using LocalRegression
using Gadfly
using Splines

# Reproduces Figure 6.1 in ESLii showing simulated differences between a nearest-neighbor kernel versus
# an Epanechnikov Kernel with Nadaraya-Watson smoothing
function figure_6_1()
    # Generate the data
    X = rand(100)
    y = sin(4*X) + .33*randn(100)

    # Generate the kernels
    nearest_30(x0, x) = LocalRegression.k_nearest(x0, x, X, 30)
    nearest_30_average(x0) = LocalRegression.NWKernel(x0, X, y, nearest_30)
    e_2(x0, x) = LocalRegression.Epanechnikov(x0, x, 0.2)
    e_2_kernel(x0) = LocalRegression.NWKernel(x0, X, y, e_2)

    # Plot the fits
    p1 = Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                     layer([x -> sin(4*x), nearest_30_average], 0.0, 1.0),
                     Guide.title("Nearest-Neighbor Kernel"))
    p2 = Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                     layer([x -> sin(4*x), e_2_kernel], 0.0, 1.0),
                     Guide.title("Epanechnikov Kernel"))
    Gadfly.draw(SVG("Figure_6_1.svg", 12inch, 6inch), hstack(p1, p2))
end

# Reproduces Figure 6.3 in ESLii showing differences between bias near the boundary for a NW-kernel versus
# local regression
function figure_6_3()
    # Generate the data
    X = rand(100)
    y = sin(4*X) + .33*randn(100)

    # Generate the kernels
    e_2(x0, x) = LocalRegression.Epanechnikov(x0, x, 0.2)
    e_2_kernel(x0) = LocalRegression.NWKernel(x0, X, y, e_2)
    llr(x0) = LocalRegression.lwpr(x0, X, y, e_2, 1)

    p1 = Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                     layer([x -> sin(4*x), e_2_kernel], 0.0, 1.0),
                     Guide.title("N-W Kernel at Boundary"))
    p2 = Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                     layer([x -> sin(4*x), llr], 0.0, 1.0),
                     Guide.title("Local Linear Regression at Boundary"))
    Gadfly.draw(SVG("Figure_6_3.svg", 12inch, 6inch), hstack(p1, p2))
end

# Reproduces Figure 6.5 in ESLii showing differences in bias in the interior for local linear versus quadratic
# regression
function figure_6_5()
    #Generate the data
    X = rand(100)
    y = sin(4*X) + .33*randn(100)

    # Generate the local regression fits
    e_3(x0, x) = LocalRegression.Epanechnikov(x0, x, 0.3)
    llr(x0) = LocalRegression.lwpr(x0, X, y, e_3, 1)
    lqr(x0) = LocalRegression.lwpr(x0, X, y, e_3, 2)
    p1 = Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                     layer([x -> sin(4*x), llr], 0.0, 1.0),
                     Guide.title("Local Linear in Interior"))
    p2 = Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                     layer([x -> sin(4*x), lqr], 0.0, 1.0),
                     Guide.title("Local Quadratic in Interior"))
    Gadfly.draw(SVG("Figure_6_5.svg", 12inch, 6inch), hstack(p1, p2))
end

# Reproduces Figure 6.6 in ESLii showing the variance functions for local polynomial regressions of varying 
# degree using a metric bandwidth tri-cube kernel
function figure_6_6()
    X = rand(100)
    tri_cube_2(x0, x) = LocalRegression.tri_cube_kernel(x0, x, 0.2)
    lcr_l(x0) = norm(LocalRegression.lwpr_l(x0, X, tri_cube_2, 0))^2
    llr_l(x0) = norm(LocalRegression.lwpr_l(x0, X, tri_cube_2, 1))^2
    lqr_l(x0) = norm(LocalRegression.lwpr_l(x0, X, tri_cube_2, 2))^2
    Gadfly.plot([lcr_l, llr_l, lqr_l], 0.0, 1.0)
end

# TODO: Can't implement until I find a good 3D plotting library
function figure_6_8()
end

# Replicates Figure 6.9 in ESLii showing local linear regression fits of ozone versus radiation
# conditioned on the wind and temperature lying in certain ranges. Note this doesn't match exactly
# because the book doesn't specify what kernel they use for weighting and on an aesthetic level
# because Gadfly currently doesn't support trellis plots
function figure_6_9()
    ozone = ESLii.read_ozone_data()
    wind_quantiles = quantile(ozone[:wind], [0, .2, .4, .6, .8, 1.0])
    temp_quantiles = quantile(ozone[:temperature], [0, .2, .4, .6, .8, 1.0])
    
    plots = Gadfly.Plot[]
    for i in 1:4
        wind_cond = wind_quantiles[i] .<= ozone[:wind] .<= wind_quantiles[i+2]
        for j in 1:4
            temp_cond = temp_quantiles[j] .<= ozone[:temperature] .<= temp_quantiles[j+2]
            sub_data = ozone[wind_cond & temp_cond, :]
            y = convert(Vector{Float64}, sub_data[:, :ozone]).^(.33)
            X = convert(Vector{Float64}, sub_data[:, :radiation])
            kernel(x0, x) = pdf(Normal(0, 1), (x - x0)/std(X))
            llr(x0) = LocalRegression.lwpr(x0, X, y, kernel, 1)
            push!(plots, Gadfly.plot(layer(x=X, y=y, Geom.point),
                                     layer(x=X, y=[llr(x) for x in X], Geom.line,
                                           Theme(default_color=color("black")))))
        end
    end
    draw(SVG("Figure_6_9.svg", 8inch, 8inch), vstack(hstack(plots[13], plots[14], plots[15], plots[16]),
                                                     hstack(plots[9], plots[10], plots[11], plots[12]),
                                                     hstack(plots[5], plots[6], plots[7], plots[8]),
                                                     hstack(plots[1], plots[2], plots[3], plots[4])))
end

# TODO: Not sure where this data comes from. It doesn't appear to be on their website
function figure_6_10()
end

# Reproduces Figure 6.12 in ESLii showing probability of CHD as a function of two risk factors in SA heart
# disease data. The model is fit by local loglinear classification. The text doesn't specify what kernel
# is used so the images diverge slightly but do show the major structure discussed in the text.
function figure_6_12()
    X, y = ESLii.read_sa_heart_data()
    sbp = X[:, 1]
    obesity = X[:, 5]

    sbp_kernel(x0, x) = LocalRegression.tri_cube_kernel(x0, x, LocalRegression.k_nearest_width(x0, sbp, 300))
    sbp_prob(x0) = LocalRegression.lllr(x0, sbp, y, sbp_kernel)
    p1 = Gadfly.plot([sbp_prob], minimum(sbp), maximum(sbp),
                Theme(default_color=color("green")),
                Guide.xlabel("Systolic Blood Pressure"), Guide.ylabel("Prevalence CHD"))

    obesity_kernel(x0, x) = LocalRegression.tri_cube_kernel(x0, x, LocalRegression.k_nearest_width(x0, obesity, 300))
    obesity_prob(x0) = LocalRegression.lllr(x0, obesity, y, obesity_kernel)
    p2 = Gadfly.plot([obesity_prob], minimum(obesity), maximum(obesity),
                Theme(default_color=color("green")),
                Guide.xlabel("Obesity"), Guide.ylabel("Prevalence CHD"))
                draw(SVG("Figure_6_12.svg", 6inch, 3inch), hstack(p1, p2))
end

# Reproduces Figure 6.13 in ESLii showing a Parzen density estimate for sbp using a Gaussian kernel. The width
# of the neighborhood isn't specified so the match isn't exact.
function figure_6_13_and_14()
    X, y = ESLii.read_sa_heart_data()
    sbp = X[:, 1]
    sbp_no_chd = X[y .== 0, 1]
    sbp_chd = X[y .== 1, 1]
    prior = length(sbp_chd)/convert(Float64, length(sbp))

    chd_lambda = std(sbp_chd)/4
    chd_kernel(x0, x) = pdf(Normal(0, 1), (x0 - x)/chd_lambda)
    chd_kde(x0) = LocalRegression.parzen_kde(x0, sbp_chd, chd_kernel, chd_lambda)
    no_chd_lambda = std(sbp_no_chd)/4
    no_chd_kernel(x0, x) = pdf(Normal(0, 1), (x0 - x)/no_chd_lambda)
    no_chd_kde(x0) = LocalRegression.parzen_kde(x0, sbp_no_chd, no_chd_kernel, no_chd_lambda)
    posterior(x0) = prior*chd_kde(x0)/((1-prior)*no_chd_kde(x0) + prior*chd_kde(x0))
    p1 = Gadfly.plot([chd_kde, no_chd_kde], minimum(sbp), maximum(sbp),
                Guide.xlabel("Systolic Blood Pressure"),
                Guide.ylabel("Density Estimate"))
    p2 = Gadfly.plot([posterior], minimum(sbp), maximum(sbp),
                Theme(default_color=color("purple")),
                Guide.xlabel("Systolic Blood Pressure"),
                Guide.ylabel("Posterior Estimate"))
    draw(SVG("Figure_6_14.svg", 8inch, 4inch), hstack(p1, p2))
end

end
