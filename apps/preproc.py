import csv
import sys
import numpy as np
import pandas as pd
from PIL import Image

def get_data(file_name, is_train):
    
    file_path = '/assets/'
    csv_file = open(file_path + file_name, newline='', encoding='utf-8')
    movie = csv.reader(csv_file, delimiter=',')
   
    # Data list used in prediction
    budget_list = []                # C
    genre_list = []                 # D
    ori_lang_list = []              # H
    overview_list = []              # J
    popularity_list = []            # K
    poster_list = []                # L
    product_country_list = []       # N
    release_date_list = []          # O
    revenue_list = []               # P
    runtime_list = []               # Q
    spok_lang_list = []             # R
    vote_avg_list = []              # W
    vot_count_list = []             # X
    
    for data in movie:
        budget_list.append(get_budget(data))
        genre_list.append(get_genre(data))
        ori_lang_list.append(get_ori_lang(data))
        overview_list.append(get_overview(data))
        popularity_list.append(get_popularity(data))
        poster_list.append(get_poster_list(data))
        product_country_list.append(get_product_country(data))
        release_date_list.append(get_release_date(data))

    csv_file.close()

    return 

