#!/usr/local/bin/python

import pandas

DATA_DIR = "../data"
PROSTATE_DATA = "prostate.data"
VOWELS_TRAIN = "vowels.train"
VOWELS_TEST = "vowels.test"
SA_HEART_DATA = "SAheart.data"
PHONEME_DATA = "phoneme.data"
BONE_DATA = "bone.data"


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


def read_prostate_data():
    """Reads the prostate data from file into a DataFrame
    """
    data_file = DATA_DIR + '/' + PROSTATE_DATA
    df = pandas.read_table(data_file, delim_whitespace=True)
    df.drop(u"row_names", axis=1, inplace=True)
    return df

def read_vowel_data(train=True):
    """Reads the vowel data from file into a DataFrame
    """
    data_file = DATA_DIR + '/' + (VOWELS_TRAIN if train else VOWELS_TEST)
    df = pandas.read_table(data_file, sep=',', header=0)
    df.drop(u"row.names", axis=1, inplace=True)
    return df


def read_sa_heart_data():
    """Reads the south african heart data from file in a DataFrame
    """
    df = pandas.read_table(DATA_DIR + '/' + SA_HEART_DATA, sep=',', header=0)
    df.drop(u"row.names", axis=1, inplace=True)
    return df


def read_phoneme_data():
    """Reads the phoneme data from file in a DataFrame
    """
    df = pandas.read_table(DATA_DIR + '/' + PHONEME_DATA, sep=',', header=0)
    df.drop(u"row.names", axis=1, inplace=True)
    return df


def read_bone_data():
    """Reads the prostate data from file into a DataFrame
    """
    return pandas.read_table(DATA_DIR + '/' + BONE_DATA,
                             delim_whitespace=True)
