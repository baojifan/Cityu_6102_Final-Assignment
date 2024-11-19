import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Import required functions from split_utils.py
from split_utils import min_rating_filter_pandas, split_pandas_data_with_ratios

# Log each step and output
print("Starting data loading...")

# Load movie data with ISO-8859-1 encoding
print("Loading movies.dat file...")
movies = pd.read_csv('movies.dat', sep='::', header=None, engine='python', encoding='ISO-8859-1')
movies.columns = ['MovieID', 'Title', 'Genres']
print("movies.dat loaded successfully.")

# Load user data with UTF-8 encoding
print("Loading users.dat file...")
users = pd.read_csv('users.dat', sep='::', header=None, engine='python', encoding='UTF-8')
users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
print("users.dat loaded successfully.")

# Load rating data with UTF-8 encoding
print("Loading ratings.dat file...")
ratings = pd.read_csv('ratings.dat', sep='::', header=None, engine='python', encoding='UTF-8')
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
print("ratings.dat loaded successfully.")

print("Data preprocessing...")

# Data preprocessing: filter users and items with a minimum number of ratings
print("Filtering users and items with minimum number of ratings...")
ratings_filtered = min_rating_filter_pandas(ratings, min_rating=5, filter_by='user', col_user='UserID', col_item='MovieID')
ratings_filtered = min_rating_filter_pandas(ratings_filtered, min_rating=5, filter_by='item', col_user='UserID', col_item='MovieID')

# Optional: Merge datasets if movie or user info is needed
# Use the filtered data
data = pd.merge(pd.merge(ratings_filtered, users), movies)

print("Splitting data into training and test sets...")

# Split the dataset into training and test sets using the function from split_utils.py
train_data, test_data = split_pandas_data_with_ratios(ratings_filtered, ratios=[0.8, 0.2], seed=42, shuffle=True)

# Since split_pandas_data_with_ratios returns a list, extract the first and second elements
train_data = train_data.copy()
test_data = test_data.copy()

# Remove any 'split_index' column if present
if 'split_index' in train_data.columns:
    train_data = train_data.drop(columns=['split_index'])
if 'split_index' in test_data.columns:
    test_data = test_data.drop(columns=['split_index'])

print("Creating user-item rating matrix...")

# Create user-item rating matrix
train_user_item_matrix = train_data.pivot(index='UserID', columns='MovieID', values='Rating')

print("Calculating user similarity matrix using cosine similarity...")

# Calculate user similarity matrix using cosine similarity
# Fill missing values with 0
train_user_item_matrix_fillna = train_user_item_matrix.fillna(0)
user_similarity = cosine_similarity(train_user_item_matrix_fillna)
user_similarity_df = pd.DataFrame(user_similarity, index=train_user_item_matrix.index, columns=train_user_item_matrix.index)

print("Defining prediction function...")

# Define a prediction function
def predict(user_id, item_id, k=20):
    # If the user is not in the similarity matrix, return the global average rating
    if user_id not in user_similarity_df.index:
        return train_data['Rating'].mean()
    # If the item is not in the item list, return the user's average rating
    if item_id not in train_user_item_matrix.columns:
        user_mean = train_user_item_matrix.loc[user_id].mean()
        if np.isnan(user_mean):
            return train_data['Rating'].mean()
        else:
            return user_mean
    # Get user similarity ranking
    sim_users = user_similarity_df[user_id].drop(user_id).dropna()
    sim_users = sim_users[sim_users > 0]  # Consider only users with positive similarity
    sim_users = sim_users.sort_values(ascending=False)
    # If no similar users, return the user's average rating
    if sim_users.empty:
        user_mean = train_user_item_matrix.loc[user_id].mean()
        if np.isnan(user_mean):
            return train_data['Rating'].mean()
        else:
            return user_mean
    # Select the top k similar users
    top_k_users = sim_users.iloc[:k]
    # Get these users' ratings for the target item
    top_k_ratings = train_user_item_matrix.loc[top_k_users.index, item_id]
    # Calculate the weighted average rating
    top_k_sim = top_k_users[top_k_ratings.notnull()]
    top_k_ratings = top_k_ratings[top_k_ratings.notnull()]
    if top_k_sim.sum() == 0:
        user_mean = train_user_item_matrix.loc[user_id].mean()
        if np.isnan(user_mean):
            return train_data['Rating'].mean()
        else:
            return user_mean
    else:
        return np.dot(top_k_sim, top_k_ratings) / top_k_sim.sum()

print("Predicting ratings for the test set...")

# Predict ratings for the test set
test_data['Prediction'] = test_data.apply(lambda x: predict(x['UserID'], x['MovieID']), axis=1)

print("Predictions completed. Calculating evaluation metrics...")

# Calculate evaluation metrics

# MAE
mae = mean_absolute_error(test_data['Rating'], test_data['Prediction'])
print(f"MAE: {mae}")

# RMSE
rmse = np.sqrt(mean_squared_error(test_data['Rating'], test_data['Prediction']))
print(f"RMSE: {rmse}")

# Convert to binary classification (e.g., ratings >= 4 are considered positive)
threshold = 4
test_data['Actual'] = (test_data['Rating'] >= threshold).astype(int)
test_data['Predicted'] = (test_data['Prediction'] >= threshold).astype(int)

# Precision, Recall, F1-Score
precision = precision_score(test_data['Actual'], test_data['Predicted'], zero_division=0)
recall = recall_score(test_data['Actual'], test_data['Predicted'], zero_division=0)
f1 = f1_score(test_data['Actual'], test_data['Predicted'], zero_division=0)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

print("Evaluation metrics calculated.")

print("Visualizing the results...")

# Visualize results in a bar chart
metrics = {'MAE': mae, 'RMSE': rmse, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(10, 6))
plt.bar(metric_names, metric_values, color='skyblue')
plt.title('User-CF Evaluation Metrics Results')
plt.ylabel('Value')
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
plt.show()

print("Visualization completed.")