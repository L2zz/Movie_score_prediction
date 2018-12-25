import csv
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import minmax_scale

# Input column: C, H, J, K, L, N, O, P, Q, R
# Output column: W, X
def get_data(file_name, is_train):

    file_path = '/assets/'
    df = pd.read_csv(file_path + file_name, encoding='utf-8', \
                     names=['A','B','C','D','E','F','G','H', \
                            'I','J','K','L','M','N','O','P', \
                            'Q','R','S','T','U','V','W','X'])
    df = df.drop(['A','B','D','E','F','G','I','M','S','T','U','V'], 1)

    # Feature Scaling
    # C: budget |  P: revenue | Q: runtime
    df[['C','P','Q']] = df[['C','P','Q']].apply(minmax_scale)

    # One-hot Encoding
    
    

    print(df)
    return X, Y

get_data('movie_test.csv', True)
