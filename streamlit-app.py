#!pip install surprise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import pickle
import streamlit as st
import NetflixLoadData as NetflixLoadData
import MovieCustomerInformation as information
max_rating = 4
min_rating = 4
#sns.set_style("darkgrid")

data_movies, data_rating, data_rating_plus_movie_title, _ = NetflixLoadData.get_data_files(use_small_dataset=True)

st.title('Netflix recommendation')
#st.write(data_rating_and_movie)
customer_id = st.selectbox('Pick a user',(532439, 588344, 596533, 609556, 607980))

st.write('You selected customer:', customer_id)
st.dataframe(information.all_id_rows(df=data_rating_plus_movie_title, type="customer_id", item_id=customer_id))

customer_movie_rated_count = information.customer_average_ratings(df=data_rating, type='customer_id', customer_id=customer_id)['rating']['count'].values[0]
customer_movie_avg_rating = round(information.customer_average_ratings(df=data_rating, type='customer_id', customer_id=customer_id)['avg_rating'].values[0], 2)

st.write("His average rating is", customer_movie_avg_rating, "from a total of", customer_movie_rated_count, "movies/tv shows")
