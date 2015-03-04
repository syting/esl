module ESLii

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

end
