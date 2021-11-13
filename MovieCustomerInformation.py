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

## id rows
def all_id_rows(df, type, item_id):
    return df[df[type] == item_id]

def all_movie_id_rows(df, movie_id):
    return data_rating[data_rating['movie_id'] == movie_id]

def all_custumers_id_rows(customer_id):
    return data_rating[data_rating['customer_id'] == customer_id]

def all_customers_id_plus_movie_title_rows(customer_id):
    return data_rating_plus_movie_title[data_rating_plus_movie_title['customer_id'] == customer_id]


### average_ratings
def all_average_ratings(df, type='movie_id'):
    ratings_stats = df.groupby(type).agg({'rating': ['sum', 'count']}).reset_index()
    ratings_stats['avg_rating'] =  ratings_stats['rating']['sum'] / ratings_stats['rating']['count']
    return ratings_stats

def all_movies_get_average_rating():
    movie_ratings_stats = data_rating.groupby('movie_id').agg({'rating': ['sum', 'count']}).reset_index()
    movie_ratings_stats['avg_rating'] =  movie_ratings_stats['rating']['sum'] / movie_ratings_stats['rating']['count']
    return movie_ratings_stats

def all_customers_get_average_rating():
    customers_ratings_stats = data_rating.groupby('customer_id').agg({'rating': ['sum', 'count']}).reset_index()
    customers_ratings_stats['avg_rating'] =  customers_ratings_stats['rating']['sum'] / customers_ratings_stats['rating']['count']
    return customers_ratings_stats



### rated_count
def all_get_rated_count(df, type):
    return df.groupby(type).agg({'movie_id': 'count'}).reset_index()

def all_movies_get_rated_count():
    return data_rating.groupby('movie_id').agg({'movie_id': 'count'}).reset_index()

def all_customers_get_movie_rated_count():
    return data_rating.groupby('customer_id').agg({'movie_id': 'count'}).reset_index()



###
def get_avg_rating_less_than(df, max_rating):
    print(df[df['avg_rating'] < max_rating])

def get_customers_avg_rating_less_than(max_rating):
    print(all_customers_average_ratings[all_customers_average_ratings['avg_rating'] < max_rating])

def get_movies_avg_rating_less_than(max_rating):
    print(all_movies_average_rating[all_movies_average_rating['avg_rating'] < max_rating])

##
def get_avg_rating_higher_than(df, min_rating):
    print(df[df['avg_rating'] > min_rating])

def get_customers_avg_rating_higher_than(min_rating):
    print(all_customers_average_ratings[all_customers_average_ratings['avg_rating'] > min_rating])

def get_movies_avg_rating_higher_than(min_rating):
    print(all_movies_average_rating[all_movies_average_rating['avg_rating'] > min_rating])




###
def get_item_avg_rating(df, type, id):
    return df[df[type] == id]

def get_movie_avg_rating(movie_id):
    output = all_movies_average_rating[all_movies_average_rating['movie_id'] == movie_id]
    #print(output)
    return output['avg_rating']

def get_users_avg_rating(customer_id):
    output = all_customers_average_ratings[all_customers_average_ratings['customer_id'] == customer_id]
    #print(output)
    return output['avg_rating']



###
def get_movies_customer_rated_higher_than(df, customer_id, min_rating=4): 
    #df is equal to data_rating_plus_movie_title
    temp = df.copy()
    return temp[(temp['customer_id'] == customer_id) & (temp['rating'] >= min_rating)].set_index('movie_id')

def get_movies_customer_rated_lower_than(df, customer_id, max_rating=4):
    #df is equal to data_rating_plus_movie_title
    temp = df.copy()
    return temp[(temp['customer_id'] == customer_id) & (temp['rating'] < max_rating)].set_index('movie_id')



###
def display_movies_customer_rated_higher_than(df, customer_id, min_rating=4):
    #df is equal to data_rating_plus_movie_title
    df_customer_liked = get_movies_customer_rated_higher_than(df=df, customer_id=customer_id, min_rating=min_rating)
    print(df_customer_liked[['movie_title', 'rating']])
    print('average rating', get_users_avg_rating(customer_id))

def display_movies_customer_rated_lower_than(df, customer_id, max_rating=4):
    #df is equal to data_rating_plus_movie_title
    df_customer_disliked = get_movies_customer_rated_lower_than(df, customer_id=customer_id, max_rating=max_rating)
    print(df_customer_disliked[['movie_title', 'rating']])
    print('average rating', get_users_avg_rating(customer_id))




def get_users_loved_hated_movies(df, customer_id, minmax_rating):
    users_ratings_higher_than_four = get_movies_customer_rated_higher_than(df, customer_id=customer_id, min_rating=minmax_rating)
    users_ratings_lower_than_four = get_movies_customer_rated_lower_than(df, customer_id=customer_id, max_rating=minmax_rating)
    print("User", customer_id ,"loved these movies")
    for rating in users_ratings_higher_than_four['movie_title']:
        print(rating)
    print('')
    print("and disliked these movies")
    for rating in users_ratings_lower_than_four['movie_title']:
        print(rating)