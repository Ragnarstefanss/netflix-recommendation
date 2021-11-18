#!pip install surprise
from os import write
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, accuracy, dataset
from surprise.model_selection import train_test_split, LeaveOneOut, cross_validate
import pickle
import streamlit as st
import NetflixLoadData as NetflixLoadData
import MovieCustomerInformation as information

max_rating = 4
min_rating = 4
reader = Reader(line_format='user item rating', rating_scale=(1, 5))

## Convert down to raw ratings
class convert_to_raw_ratings(dataset.DatasetAutoFolds):
    def __init__(self, df, reader):
        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in zip(df['customer_id'], df['movie_id'], df['rating'])]
        self.reader=reader

def load_pickle(name):
    path_name = "pickle/"+name+".pickle"
    return_input = open(path_name, "rb")
    return pickle.load(return_input)

## helper functions
def get_drop_list(type="movie_id"):
    #>IF MOVIE:  movie_id, movie rating count, movie rating mean
    #>IF CUSTOMER: customer_id, custumer rating count, customer rating mean
    df_count_mean_summary = data_rating.groupby(type)['rating'].agg(['count', 'mean'])
    df_count_mean_summary.index = df_count_mean_summary.index.map(int)
    #>IF MOVIE: returns 1799.0 as a benchmark number
    #IF CUSTOMER: returns 52.0 as a benchmark number
    benchmark = round(df_count_mean_summary['count'].quantile(0.7),0)
    # drop all rows below benchmark
    df_drop_list = df_count_mean_summary[df_count_mean_summary['count'] < benchmark]
    # return all indexes to drop
    return df_drop_list

def get_customer_recommendations(customer_id, predictor):
    # drop movies/tv shows below benchmark (threshold)
    df_movie_drop_list = get_drop_list(type="movie_id")
    #> returns movie_id, movie_year, movie_title
    chosen_customer_pred = data_movies.copy()
    # fails if movie_id is the index so we have to reset the index back to normal (0-N)
    chosen_customer_pred = chosen_customer_pred.reset_index()
    # makes sure that we only pick movies that are not in the movie dropped list
    chosen_customer_pred = chosen_customer_pred[~chosen_customer_pred['movie_id'].isin(df_movie_drop_list)]
    # make prediction for customer with id = <customer_id> and put it into 'estimated_score'
    chosen_customer_pred['estimated_score'] = chosen_customer_pred['movie_id'].apply(lambda x: predictor.predict(customer_id, x).est)
    # sort by 'estimated score'
    chosen_customer_pred = chosen_customer_pred.sort_values('estimated_score', ascending=False).set_index('movie_id')
    return chosen_customer_pred

def display_customers_recommendations(df=[], number_to_show=20):
    print("Movies/TV Shows recommended to customer")
    tmp_df = df[['movie_title', 'estimated_score']][0:number_to_show]
    tmp_df = tmp_df.set_index('movie_title')
    print(tmp_df)


def display_recommendation(customer_id, number_to_show, predictor):
    chosen_customer_pred = get_customer_recommendations(customer_id=customer_id, predictor=predictor)
    display_customers_recommendations(df=chosen_customer_pred, number_to_show=number_to_show)

def return_recommendation(customer_id, number_to_show, predictor):
    chosen_customer_pred = get_customer_recommendations(customer_id=customer_id, predictor=predictor)
    tmp_df = chosen_customer_pred[['movie_title', 'estimated_score']][0:number_to_show]
    tmp_df = tmp_df.set_index('movie_title')
    return tmp_df
    

def show_evaluation(predictor, dataset):
    return cross_validate(predictor, dataset, measures=['MSE', 'RMSE', 'MAE'], cv=5, verbose=True)

def print_evaluation_accuracy(prediction):
    st.write("\n Accuracy of the model")
    st.write("RMSE: ", accuracy.rmse(prediction, verbose=False))
    st.write("MSE: ", accuracy.mse(prediction, verbose=False))
    st.write("MAE: ", accuracy.mae(prediction, verbose=False))
    # FCP = Fraction of Concordant Pairs
    st.write("FCP: ", accuracy.mae(prediction, verbose=False))

# streamlit functions

def write_subheader(title):
    st.markdown('#')
    st.subheader(title)

def get_user_stats(customer_id):
    # stats to use 
    ## movie count
    all_customers_movie_counts = information.all_average_ratings(df=data_rating, type='customer_id')["rating"]["count"].mean()
    movie_count_change = ((customer_movie_rated_count - all_customers_movie_counts) / all_customers_movie_counts)*100
    movie_count_delta = round(movie_count_change, 2)
    movie_count_delta_string = str(movie_count_delta)+"% (avg is " + str(round(all_customers_movie_counts)) + ")"

    ## customer average rating
    all_customers_average_ratings =  information.all_average_ratings(df=data_rating, type='customer_id')["avg_rating"].mean()
    avg_rating_change = ((customer_movie_avg_rating - all_customers_average_ratings) / all_customers_average_ratings)*100
    avg_rating_delta = round(avg_rating_change, 2)
    avg_rating_delta_string = str(avg_rating_delta)+"% (avg is " + str(round(all_customers_average_ratings)) + ")"

   
    # write to website
    st.markdown('#')
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Movies rated", value=int(customer_movie_rated_count), delta=movie_count_delta_string)
    col2.metric(label="AVG Rating", value=customer_movie_avg_rating, delta=avg_rating_delta_string)
    col3.metric(label="AVG Year Movie Ratings", value="86%", delta="xx%  compared to average")


def call_algo():
    trainSet = load_pickle("algo_trainSet")
    testSet = load_pickle("algo_testSet")
    algo = load_pickle("algo_svd")
    st.dataframe(return_recommendation(customer_id=customer_id, number_to_show=9, predictor=algo))
    algo_predictions = algo.test(testSet)
    print_evaluation_accuracy(algo_predictions)


data_movies, data_rating, data_rating_plus_movie_title, _ = NetflixLoadData.get_data_files(use_small_dataset=True)
dataset = Dataset.load_from_df(data_rating[['customer_id', 'movie_id', 'rating']], reader)
trainset = dataset.build_full_trainset()


st.title('Netflix recommendation')
#st.write(data_rating_and_movie)
customer_id = st.selectbox('Pick a user',(532439, 588344, 596533, 609556, 607980))
customers_all_ratings = information.all_id_rows(df=data_rating_plus_movie_title, type="customer_id", item_id=customer_id)
customer_movie_rated_count = information.customer_average_ratings(df=data_rating, type='customer_id', customer_id=customer_id)['rating']['count'].values[0]
customer_movie_avg_rating = round(information.customer_average_ratings(df=data_rating, type='customer_id', customer_id=customer_id)['avg_rating'].values[0], 2)


get_user_stats(customer_id)

write_subheader("Movies/TV Shows rated by user")
st.dataframe(customers_all_ratings)

write_subheader("Recommended movies to user")
print("\nBuilding recommendation model...")
genre = st.selectbox("Pick model", ('algo', 'svd', 'big'))

if(genre == "algo"):
    call_algo()
if(genre == "svd"):
    call_alternative()


#0 : 1452669
#1 : 4227
#2 : 2
#algo_predictions
#+
#3: 2.4164138660128724
#4 : {"was_impossible": false}