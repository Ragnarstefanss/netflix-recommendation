#!pip install surprise
#!pip install import-ipynb
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
data_movies, data_rating, data_rating_plus_movie_title, _ = NetflixLoadData.get_data_files(use_small_dataset=True)

def all_id_rows(df, type, item_id):
    return df[df[type] == item_id]

def all_average_ratings(df, type='movie_id'):
    ratings_stats = df.groupby(type).agg({'rating': ['sum', 'count']}).reset_index()
    ratings_stats['avg_rating'] =  ratings_stats['rating']['sum'] / ratings_stats['rating']['count']
    return ratings_stats

def customer_average_ratings(df, type='customer_id', customer_id=0):
    ratings_stats = df[df[type]==customer_id].groupby(type).agg({'rating': ['sum', 'count']}).reset_index()
    ratings_stats['avg_rating'] =  ratings_stats['rating']['sum'] / ratings_stats['rating']['count']
    return ratings_stats

def all_get_rated_count(df, type):
    return df.groupby(type).agg({'movie_id': 'count'}).reset_index()

def get_avg_rating_less_than(df, max_rating):
    return df[df['avg_rating'] < max_rating]

def get_avg_rating_higher_than(df, min_rating):
    return df[df['avg_rating'] > min_rating]

def get_item_avg_rating(df, type, item_id):
    return df[df[type] == item_id]


def get_movies_customer_rated_higher_than(df, customer_id, min_rating=4): 
    #df is equal to data_rating_plus_movie_title
    temp = df.copy()
    return temp[(temp['customer_id'] == customer_id) & (temp['rating'] >= min_rating)].set_index('movie_id')

def get_movies_customer_rated_lower_than(df, customer_id, max_rating=4):
    #df is equal to data_rating_plus_movie_title
    temp = df.copy()
    return temp[(temp['customer_id'] == customer_id) & (temp['rating'] < max_rating)].set_index('movie_id')


def display_movies_customer_rated_higher_than(df, customer_id, min_rating=4):
    #df is equal to data_rating_plus_movie_title
    df_customer_liked = get_movies_customer_rated_higher_than(df=df, customer_id=customer_id, min_rating=min_rating)
    print(df_customer_liked[['movie_title', 'rating']])
    customers_ratings_stats = df.groupby('customer_id').agg({'rating': ['sum', 'count']}).reset_index()
    customers_ratings_stats['avg_rating'] =  customers_ratings_stats['rating']['sum'] / customers_ratings_stats['rating']['count']
    print('average rating', customers_ratings_stats[customers_ratings_stats['customer_id'] == customer_id]['avg_rating'])

def display_movies_customer_rated_lower_than(df, customer_id, max_rating=4):
    #df is equal to data_rating_plus_movie_title
    df_customer_disliked = get_movies_customer_rated_lower_than(df, customer_id=customer_id, max_rating=max_rating)
    print(df_customer_disliked[['movie_title', 'rating']])
    customers_ratings_stats = df.groupby('customer_id').agg({'rating': ['sum', 'count']}).reset_index()
    customers_ratings_stats['avg_rating'] =  customers_ratings_stats['rating']['sum'] / customers_ratings_stats['rating']['count']
    print('average rating', customers_ratings_stats[customers_ratings_stats['customer_id'] == customer_id]['avg_rating'])


def get_users_loved_hated_movies(df, customer_id, minmax_rating):
    users_ratings_higher_than_four = get_movies_customer_rated_higher_than(df=df, customer_id=customer_id, min_rating=minmax_rating)
    users_ratings_lower_than_four = get_movies_customer_rated_lower_than(df=df, customer_id=customer_id, max_rating=minmax_rating)
    print("User", customer_id ,"loved these movies")
    for rating in users_ratings_higher_than_four['movie_title']:
        print(rating)
    print('')
    print("and disliked these movies")
    for rating in users_ratings_lower_than_four['movie_title']:
        print(rating)

def get_customers_who_rated_movie_title(df=data_rating_plus_movie_title, movie_title=""):
    return df[df["movie_title"] == movie_title]

def get_avg_rating_for_movie_title(df=data_movies, movie_title=""):
    tmp_data_movies = df[df["movie_title"] == movie_title]
    tmp_movie_id = tmp_data_movies['movie_id'].values[0]
    tmp_data_movies.columns = pd.MultiIndex.from_product([['movie_id'], tmp_data_movies.columns])
    all_movies_average_rating = all_average_ratings(df=data_rating, type='movie_id')
    return get_item_avg_rating(df=all_movies_average_rating , type='movie_id', item_id=tmp_movie_id).join(tmp_data_movies)