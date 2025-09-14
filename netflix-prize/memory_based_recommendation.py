### Basic Lib for Machine Learning
import numpy as np
import pandas as pd

### Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

### For matrix factorization (SVD)
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

import warnings
warnings.filterwarnings('ignore')

import csv

titles = []
with open('data/netflix-prize-data/movie_titles.csv', encoding='latin-1') as f:
    reader = csv.reader(f)
    for row in reader:
        try:
            movie_id = int(row[0])
            year = int(row[1])
            title = ",".join(row[2:])
            titles.append([movie_id, year, title])
        except:
            continue

titles_df = pd.DataFrame(titles, columns=['movie_id', 'year', 'title'])

ratings_list = []
with open('data/netflix-prize-data/combined_data_1.txt', 'r') as file:
    movie_id = None
    for line in file:
        line = line.strip()
        if line.endswith(':'):
            movie_id = int(line[:-1])
        else:
            user_id, rating, date = line.split(',')
            ratings_list.append([int(user_id), movie_id, int(rating), date])

ratings_df = pd.DataFrame(ratings_list, columns=['user_id','movie_id','rating','date'])

df = pd.merge(ratings_df, titles_df, on='movie_id')

print("Ratings shape:", ratings_df.shape)
print("Titles shape:", titles_df.shape)
print("Final merged shape:", df.shape)
df.head()