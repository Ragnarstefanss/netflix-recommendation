#!pip install surprise
#!pip install import-ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import pickle
#sns.set_style("darkgrid")
from surprise import accuracy
from collections import defaultdict
from surprise import KNNBasic
from collections import defaultdict
from operator import itemgetter
import heapq
import os
import csv

def create_dataframe(use_pickle=True):
    if(use_pickle == True):
        in_pickle = open("pickle/netflix_ratings.pickle", "rb")
        data = pd.DataFrame(pickle.load(in_pickle), columns=['movie_id', 'customer_id', 'rating', 'index']).drop(['index'], axis=1)
    else:
        index = 1
        last_movie_id = "1"
        new_data = []
        ## import all combined_data files to one large pandas dataframe
        ##> returns index, customer_id (which is both movie and customer), rating (NaN = customer_id is a movie, Not NaN = customers rating)
        df_all = pd.read_csv('./data/combined_data_1.txt', header = None, names = ['customer_id', 'rating'], usecols = [0,1])
        #df_all = df_all.append(pd.read_csv('./data/combined_data_2.txt', header = None, names = ['customer_id', 'rating'], usecols = [0,1]))
        #df_all = df_all.append(pd.read_csv('./data/combined_data_3.txt', header = None, names = ['customer_id', 'rating'], usecols = [0,1]))
        #df_all = df_all.append(pd.read_csv('./data/combined_data_4.txt', header = None, names = ['customer_id', 'rating'], usecols = [0,1]))
        df_all.index = np.arange(0,len(df_all))
        df_all['rating'] = df_all['rating'].astype(float)
        
        for customer_id in df_all["customer_id"]:
            # if we find : that means this is a movie_id and not customer_id
            if(customer_id.find(":") > 0):
                movie_id = customer_id.replace(":", "")
                last_movie_id = movie_id
            else:
                # we have this row index so use it to get rating
                rating = df_all["rating"][index-1]
                new_data.append([last_movie_id, customer_id, rating, index])
            index += 1
        #output to pickle file
        movies_customers_ratings = open("pickle/netflix_ratings.pickle","wb")
        pickle.dump(new_data, movies_customers_ratings)
        movies_customers_ratings.close()
        data = pd.DataFrame(new_data, columns=['movie_id', 'customer_id', 'rating', 'index']).drop(['index'], axis=1)

    # change columns to numerical
    data['movie_id'] = data['movie_id'].astype(int)
    data['customer_id'] = data['customer_id'].astype(int)
    data["rating"] = data["rating"].astype(float)
    return data

def get_dataframes(use_pickle=True):
    # dataframe containing all informations about the movies
    #> returns movie_id, movie_year, movie_title
    data_movies = pd.read_csv('./data/movie_titles.csv', header = None, names = ['movie_id', 'movie_year', 'movie_title'], usecols = [0,1,2], encoding="latin1")
    #data_movies_import.set_index('movie_id', inplace = True)

    ## ------------------------------------------------------------------------------------- ##

    # dataframe containing all informations about the movie ratings by customer
    #> returns index, movie_id, customer_id, rating
    data_rating = create_dataframe(use_pickle=use_pickle)

    # ## ------------------------------------------------------------------------------------- ##

    # ##combine customer ratings to movie titles
    # ##> returns index, movie_id, customer_id, rating, movie_year, movie_title
    data_rating_plus_movie_title = data_rating.merge(data_movies, on="movie_id", how="inner")

    ## ------------------------------------------------------------------------------------- ##

    data_movies_categorize = pd.read_csv('./data/movies.csv', header = None, names = ['movie_id', 'movie_title', 'genres'], usecols = [0,1,2], encoding="latin1")[1:] #dataset is off by one
    #data_movies_categorize.set_index('movie_id', inplace = True)
    data_movies_categorize_split = data_movies_categorize['movie_title'].str.split('(', n = 1, expand=True) # split movie_title to movie_title and movie_year
    data_movies_categorize_split[1] = data_movies_categorize_split[1].str.replace(r')', '') #removing ) at the end of movie_year
    data_movies_categorize["movie_year"] = data_movies_categorize_split[1]
    data_movies_categorize["movie_title"] = data_movies_categorize_split[0]
    #data_movies_categorize["movie_year"] = data_movies_categorize["movie_year"].astype(float)
    #data_movies_categorize
    data_movies_categorize_cleaned = data_movies_categorize[pd.to_numeric(data_movies_categorize['movie_year'], errors='coerce').notnull()]
    data_movies_categorize_cleaned["movie_year"] = data_movies_categorize_cleaned["movie_year"].astype(float)
    
    
    return data_movies, data_rating, data_rating_plus_movie_title, data_movies_categorize_cleaned

def get_sample_of_data(df1, df2, max_n=100000):
    # set max_n as something other than 0 if you want to only get part of the results (for quicker access)
    _df1 = df1[:max_n]
    _df2 = df2[:max_n]

    return _df1, _df2 