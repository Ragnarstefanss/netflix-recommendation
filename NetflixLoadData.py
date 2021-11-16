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

def get_data_files(use_small_dataset=True):
    if(use_small_dataset):
        data_rating = pd.read_csv('./data/smaller_netflix_ratings_cleaned.csv', sep=',')
        data_rating_plus_movie_title = pd.read_csv('./data/smaller_netflix_ratings_plus_movie_title_cleaned.csv', sep=',')
    else:
        data_rating = pd.read_csv('./data/netflix_ratings_cleaned.csv', sep=',')
        data_rating_plus_movie_title = pd.read_csv('./data/netflix_ratings_plus_movie_title_cleaned.csv', sep=',')
    
    movies = pd.read_csv('./data/movie_titles.csv', header = None, names = ['movie_id', 'movie_year', 'movie_title'], usecols = [0,1,2], encoding="latin1")
    other_movies_categorize = pd.read_csv('./data/netflix_movies_categorized_cleaned.csv', sep=',')
    return movies, data_rating, data_rating_plus_movie_title, other_movies_categorize

def commented_implementation():
    print("This part was a huge problem for me since the original files where weirdly structured")
    print("For example the ratings file was structured with the movie_id randomly being placed in a line and each line after that was refering to that movie without having the id in their line, then another movie randomly started in a line and lines after that were a reference to that movie")
    print("\nHow it looked before:")
    print("1:\n 1488844, 3, 2005-09-06 \n 822109, 5, 2005-05-13 \n 885013, 4, 2005-10-19 \n 30878, 4, 2005-12-26 \n....\n....\n...\n")
    print("2:\n 2059652, 4, 2005-09-05 \n 1666394, 3, 2005-04-19 \n 1759415, 4, 2005-04-22 \n 1959936, 5, 2005-11-21 \n....\n....\n...\n")
    print("the way I fixed this was include everything in a dataframe and if there was a missing rating then that was a movie_id other wise it was a customer rating, then I combined all these rows to those ratingsm so each movie had a correct reference")

def commented_code_that_was_used_before():
    ##data_movies, tmp_data_rating, tmp_data_rating_plus_movie_title, data_movies_categorized = NetflixLoadData.get_dataframes(use_pickle=use_pickle_file)
    ## save to csv file so we don't have to run everything again
    #tmp_data_rating.to_csv('data/netflix_ratings_cleaned.csv', index=False, sep=',')
    #tmp_data_rating_plus_movie_title.to_csv('data/netflix_ratings_plus_movie_title_cleaned.csv', index=False, sep=',')
    #data_movies_categorized.to_csv('data/netflix_movies_categorized_cleaned.csv', index=False, sep=',')

    ## get a smaller dataset then above (which is the entire dataset)
    #data_rating, data_rating_plus_movie_title = NetflixLoadData.get_sample_of_data(tmp_data_rating, tmp_data_rating_plus_movie_title, max_n=max_n)


    # data_rating = tmp_data_rating
    # unique_users = np.unique(data_rating['customer_id'][0:5000])
    # small_rating_dataset = data_rating[data_rating['customer_id'].isin(unique_users)]
    ## save the 800 thousand rows to file
    # small_rating_dataset.to_csv('data/smaller_netflix_ratings_cleaned.csv', index=False, sep=',')

    #data_rating_plus_movie_title = tmp_data_rating_plus_movie_title
    #small_rating_plus_movies_dataset = data_rating_plus_movie_title[data_rating_plus_movie_title['customer_id'].isin(unique_users)]
    #small_rating_plus_movies_dataset.to_csv('data/smaller_netflix_ratings_plus_movie_title_cleaned.csv', index=False, sep=',')
    print("")
