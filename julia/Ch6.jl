module Ch6

using Classification
using DataFrames, DataFramesUtil
using LocalRegression
using Gadfly
using Splines

function figure_6_1()
    X = rand(100)
    y = sin(4*X) + .33*randn(100)
    nearest_30(x0, x) = LocalRegression.k_nearest(x0, x, X, 30)
    nearest_30_average(x0) = LocalRegression.NWKernel(x0, X, y, nearest_30)
    e_2(x0, x) = LocalRegression.Epanechnikov(x0, x, 0.2)
    e_2_kernel(x0) = LocalRegression.NWKernel(x0, X, y, e_2)
    Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                layer([x -> sin(4*x), nearest_30_average, e_2_kernel], 0.0, 1.0))
end

function figure_6_3()
    X = rand(100)
    y = sin(4*X) + .33*randn(100)
    e_2(x0, x) = LocalRegression.Epanechnikov(x0, x, 0.2)
    e_2_kernel(x0) = LocalRegression.NWKernel(x0, X, y, e_2)
    llr(x0) = LocalRegression.lwpr(x0, X, y, e_2, 1)
    Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                layer([x -> sin(4*x), e_2_kernel, llr], 0.0, 1.0))
end

function figure_6_5()
    X = rand(100)
    y = sin(4*X) + .33*randn(100)
    e_3(x0, x) = LocalRegression.Epanechnikov(x0, x, 0.3)
    llr(x0) = LocalRegression.lwpr(x0, X, y, e_3, 1)
    lqr(x0) = LocalRegression.lwpr(x0, X, y, e_3, 2)
    Gadfly.plot(layer(x=X, y=y, Geom.point, Theme(default_color=color("black"))),
                layer([x -> sin(4*x), llr, lqr], 0.0, 1.0))
end

function figure_6_6()
    X = rand(100)
    tri_cube_2(x0, x) = LocalRegression.tri_cube_kernel(x0, x, 0.2)
    lcr_l(x0) = norm(LocalRegression.lwpr_l(x0, X, tri_cube_2, 0))^2
    llr_l(x0) = norm(LocalRegression.lwpr_l(x0, X, tri_cube_2, 1))^2
    lqr_l(x0) = norm(LocalRegression.lwpr_l(x0, X, tri_cube_2, 2))^2
    Gadfly.plot([lcr_l, llr_l, lqr_l], 0.0, 1.0)
end

end
