import csv
import sys
import numpy as np
import pandas as pd
from PIL import Image

def get_data(file_name, is_train):

    file_path = '/assets/'
    df = pd.read_csv(file_path + file_name, encoding='utf-8'i,\
                     names=['A','B','C','D','E','F','G','H', \
                            'I','J','K','L','M','N','O','P', \
                            'Q','R','S','T','U','V','W','X'])
    # Input: C, D, H, J, K, L, N, O, P, Q, R
    # Output: W, X
    df = df.drop(['A','B','E','F','G','I','M','S','T','U','V'], 1)

    # Feature Scaling

    # One-hot Encoding

    return df

get_data('movie_train.csv', True)
