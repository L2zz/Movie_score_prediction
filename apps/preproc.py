import csv
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import minmax_scale

# Input column: C, H, J, K, N, O, P, Q, R
# Output column: W, X
def get_data(is_train):
    # Set file path and name
    file_path = '/assets/'
    if (is_train):
        file_name = 'movie_train.csv'
    else:
        file_name = 'movie_test.csv'

    # Select data from file
    df = pd.read_csv(file_path + file_name, encoding='utf-8', \
                     names=['A','B','C','D','E','F','G','H', \
                            'I','J','K','L','M','N','O','P', \
                            'Q','R','S','T','U','V','W','X'])
    df = df.drop(['A','B','D','E','F','G','I','L','M','S','T','U','V'], 1)

    # Drop Error data
    if (is_train):
        df = df.drop(df.index[4180])

    # Get year from release date
    df['O'] = pd.to_datetime(df['O'])
    df['O'] = df['O'].dt.year

    # Feature Scaling
    # C: budget |  K: popularity | O: release_date_year | P: revenue | Q: runtime
    df[['C','K','O','P','Q']] = df[['C','K','O','P','Q']].apply(minmax_scale)

    # One-hot Encoding
    # H: original_language | N: production_countries | R: spoken_language
    df[['H','N','R']] = pd.get_dummies(df[['H','N','R']])

    print(df[['H','N','R']])
    return X, Y

def get_poster(index, batch_size, is_train):
    # Set file path and name
    file_path = '/assets/'
    if (is_train):
        poster_file_name = 'movie_train_image.npy'
    else:
        poster_file_name = 'movie_test_image.npy'

    # Get batch size posters
    posters = np.load(file_path + poster_file_name)
    batch_posters = posters[index: index+batch_size]
    print(batch_posters)
    return batch_posters

get_data(index, 100, True)
