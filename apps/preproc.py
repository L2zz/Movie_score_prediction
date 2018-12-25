import csv
import sys
import numpy as np
import pandas as pd
from PIL import Image

def get_data(file_name, is_train):

    file_path = '/assets/'
    preproc_data = pd.read_csv(file_path + file_name, encoding='utf-8')

    # Input: C, D, H, J, K, L, N, O, P, Q, R
    # Output: W, X
    # Preprocessed data list used in prediction
    # Feature Scaling

    # One-hot Encoding

    csv_file.close()

    return preproc_data
