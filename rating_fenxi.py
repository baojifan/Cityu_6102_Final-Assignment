import pandas as pd
import matplotlib.pyplot as plt
import os
# 稀疏度：95.53%
os.chdir('/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m')


column_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings = pd.read_csv('dataset/ratings.dat', sep='::', header=None, names=column_names, engine='python')


num_users = ratings['UserID'].nunique()
num_movies = ratings['MovieID'].nunique()


num_ratings = len(ratings)

total_possible_ratings = num_users * num_movies
sparsity = 1 - (num_ratings / total_possible_ratings)

print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Number of ratings: {num_ratings}")
print(f"Sparsity: {sparsity:.4f}")


plt.figure(figsize=(8, 6))


plt.hist(ratings['Rating'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor='black', rwidth=0.8)


plt.title('Distribution of Ratings in MovieLens 1M', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Count', fontsize=14)


plt.show()