#!pip install surprise
#!pip install import-ipynb
import MovieCustomerInformation as information
import os
import csv
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import pickle
from surprise import accuracy
from collections import defaultdict
from surprise import KNNBasic
from collections import defaultdict
from operator import itemgetter
import NetflixLoadData as NetflixLoadData
#sns.set_style("darkgrid")


def save_to_pickle(name, df):
    path_name = "pickle/"+name+".pickle"
    pickle_file = open(path_name, "wb")
    pickle.dump(df, pickle_file)
    pickle_file.close()


def load_pickle(name):
    path_name = "pickle/"+name+".pickle"
    return_input = open(path_name, "rb")
    return pickle.load(return_input)

use_pickle_file = True
max_n = 2500000  # how many rows we want from data_ratings and data_rating_plus_movie_title
max_rating = 4
min_rating = 4
reader = Reader(line_format='user item rating', rating_scale=(1, 5))

data_movies, tmp_data_rating, tmp_data_rating_plus_movie_title, data_movies_categorized = NetflixLoadData.get_dataframes(use_pickle=use_pickle_file)
data_rating, data_rating_plus_movie_title = NetflixLoadData.get_sample_of_data(tmp_data_rating, tmp_data_rating_plus_movie_title, max_n=max_n)

# get the average movie rating for all customers
# used to determine if this user typically gives bad or good reviews
# and then we can see if he really hates or loves a movie
all_customers_average_ratings = information.all_average_ratings(df=data_rating, type='customer_id')
print(all_customers_average_ratings)