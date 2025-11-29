import pandas as pd

df = pd.read_csv('C:/Users/Jeferson Santana/Documents/recommender_course/data/rating.csv')

# make the userId start from 0
df['userId'] = df['userId'] - 1

# create a mapping for movieId
unique_movie_ids = set(df['movieId'].values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

# add them to the dataframe
df['movie_idx'] = df.apply(lambda row: movie2idx[row['movieId']], axis=1)
df = df.drop(columns=['timestamp'])
df.to_csv('C:/Users/Jeferson Santana/Documents/recommender_course/data/edited_rating.csv')