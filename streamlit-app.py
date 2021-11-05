#!pip install surprise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import pickle
import streamlit as st
#sns.set_style("darkgrid")

pickle_file = open("pickle/small_movies_customers_ratings.pickle", "rb")
data = pd.DataFrame(pickle.load(pickle_file), columns=['movie_id', 'customer_id', 'rating', 'index']).drop(['index'], axis=1)
data = data[:10000]
data['movie_id'] = data['movie_id'].astype(int)
data['customer_id'] = data['customer_id'].astype(int)
data["rating"] = data["rating"].astype(int)

df_movies = pd.read_csv('./data/movie_titles.csv', header = None, names = ['movie_id', 'movie_year', 'movie_title'], usecols = [0,1,2], encoding="latin1")
data_rating_and_movie = data.merge(df_movies, on="movie_id", how="inner")


st.title('Netflix recommendation')
#st.write(data_rating_and_movie)
option = st.selectbox(
 'Pick a user',
(1488844, 822109, 30878))
st.write('You selected:', option)
#print(data_rating_and_movie)
display_data = data_rating_and_movie[data_rating_and_movie['customer_id'] == option]
#print(display_data)
st.dataframe(display_data)