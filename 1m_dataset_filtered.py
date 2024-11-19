import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset')
# Read the cleaned movie ratings dataset
df = pd.read_csv('ml-1m/processed_merged_data.csv')

# Movie Filtering: Calculate the number of users who rated each movie and filter out movies with fewer than 10 user ratings
movie_counts = df.groupby('movie_id').size()
filtered_movie_ids = movie_counts[movie_counts >= 10].index

# Filter the dataset, keeping only records where movie_id is in filtered_movie_ids
filtered_df = df[df['movie_id'].isin(filtered_movie_ids)]

# Plot scatter plot after movie filtering
plt.figure(figsize=(10, 4))
plt.scatter(movie_counts.index, movie_counts.values, color='green', alpha=0.6)
plt.axhline(y=10, color='red', linestyle='-')
plt.title('Number of Users Who Rated Each Movie (After Movie Filtering)', fontsize=14)
plt.xlabel('MovieId', fontsize=12)
plt.ylabel('No. of Users (voted)', fontsize=12)
plt.show()

#User Filtering: Calculate the number of ratings each user has made and filter out users who have rated fewer than 20 movies
user_counts = filtered_df.groupby('user_id').size()
filtered_user_ids = user_counts[user_counts >= 20].index

# Filter the dataset, keeping only records where user_id is in filtered_user_ids
final_filtered_df = filtered_df[filtered_df['user_id'].isin(filtered_user_ids)]

# Save to a subfolder with a relative path
# filtered_df.to_csv('ml-1m/filtered_ratings_movies.csv', index=False)

# Plot scatter plot after user filtering
plt.figure(figsize=(10, 4))
plt.scatter(user_counts.index, user_counts.values, color='blue', alpha=0.6)
plt.axhline(y=20, color='red', linestyle='-')
plt.title('Number of Movies Rated by Each User (After User Filtering)', fontsize=14)
plt.xlabel('UserId', fontsize=12)
plt.ylabel('No. of Movies (rated)', fontsize=12)
plt.show()