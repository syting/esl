module DataFramesUtil

using DataFrames

function Base.convert{T<:Number}(::Type{Array{T,2}}, df::DataFrame)
    data = Array(T, size(df))
    for i in 1:size(df,2)
        data[:,i] = df[i]
    end
    return data
end

function Base.cor(df::DataFrame)
    cor(convert(Matrix{Float64}, df)), names(df)
end

function standardize!(df::DataFrame, center=true, scale=true)
    for i in 1:size(df,2)
        if center
            df[i] -= mean(df[i])
        end
        if scale
            scale!(df[i], 1/std(df[i]))
        end
    end
    return df
end

function standardize(df::DataFrame, center=true, scale=true)
    standardized_df = copy(df)
    return standardize!(standardized_df, center=center, scale=scale)
end

end
