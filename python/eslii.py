#!/usr/local/bin/python

import pandas

DATA_DIR = "../data"
PROSTATE_DATA = "prostate.data"


def read_prostate_data():
    """Reads the prostate data from file into a DataFrame
    """
    return pandas.read_table(DATA_DIR + '/' + PROSTATE_DATA,
                             delim_whitespace=True)


def standardize_data(df, demeanCols=None, scaleCols=None):
    """Standardizes columns in the passed DataFrame df by first demeaning the
    columns in demeanCols and then scaling the columns in scaleCols to unit
    variance. By default this is done to all columns
    """
    if demeanCols is None:
        demeanCols = df.columns
    if scaleCols is None:
        scaleCols = df.columns

    df[demeanCols] -= df[demeanCols].mean()
    df[scaleCols] /= df[scaleCols].std()
    return df
