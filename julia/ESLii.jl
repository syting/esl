module ESLii

using DataFrames, DataFramesUtil

DATA_DIR = "../data"
PROSTATE_DATA = "prostate.data"
VOWELS_TRAIN = "vowels.train"
VOWELS_TEST = "vowels.test"
SA_HEART_DATA = "SAheart.data"
PHONEME_DATA = "phoneme.data"
BONE_DATA = "bone.data"

function read_prostate_data(train=true, standardize=true)
    #Reads the prostate data from file into a DataFrame
    
    df = readtable("$DATA_DIR/$PROSTATE_DATA", separator=' ', header=true)[:, 2:end]
    if standardize
        df[1:8] = DataFramesUtil.standardize!(df[1:8])
    end

    if train
        return df[df[:train] .== true, :][1:end - 1]
    else
        return df[df[:train] .== false, :][1:end - 1]
    end
end

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

function read_vowel_data(train=true)
    # Reads the vowel data from file into a DataFrame
    data_file = "$DATA_DIR/$VOWELS_TRAIN"
    if !train
        data_file = "$DATA_DIR/$VOWELS_TEST"
    end
    data_file = (train ? "$DATA_DIR/$VOWELS_TRAIN" : "$DATA_DIR/$VOWELS_TEST")
    return load_data(data_file, ',', true, :y, true, [:row_names])
end

function read_sa_heart_data()
    # Reads the south african heart data from file in a DataFrame
    return load_data(DATA_DIR + "/" + SA_HEART_DATA, ',', true, :chd, false, [:row_names, :adiposity, :typea], [:famhist])
end


end
