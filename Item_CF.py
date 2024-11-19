# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Import functions from split_utils.py
from split_utils import min_rating_filter_pandas, split_pandas_data_with_ratios

# Logging each step
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

# Filter users and items with minimum number of ratings
ratings_filtered = min_rating_filter_pandas(ratings, min_rating=5, filter_by='user', col_user='UserID', col_item='MovieID')
ratings_filtered = min_rating_filter_pandas(ratings_filtered, min_rating=5, filter_by='item', col_user='UserID', col_item='MovieID')

# Merge datasets (optional if user or movie info is needed)
data = pd.merge(pd.merge(ratings_filtered, users), movies)

print("Splitting data into training and test sets...")

# Split the dataset (e.g., 80% training and 20% testing)
train_data, test_data = split_pandas_data_with_ratios(ratings_filtered, ratios=[0.8, 0.2], seed=42, shuffle=True)

# Since split_pandas_data_with_ratios returns lists, extract the relevant elements
train_data = train_data.copy()
test_data = test_data.copy()

# Remove the 'split_index' column if it exists
if 'split_index' in train_data.columns:
    train_data = train_data.drop(columns=['split_index'])
if 'split_index' in test_data.columns:
    test_data = test_data.drop(columns=['split_index'])

print("Creating item-user rating matrix...")

# Create the item-user rating matrix
train_item_user_matrix = train_data.pivot(index='MovieID', columns='UserID', values='Rating')

print("Calculating item similarity matrix...")

# Compute the item similarity matrix using cosine similarity
# Fill missing values with 0
train_item_user_matrix_fillna = train_item_user_matrix.fillna(0)
item_similarity = cosine_similarity(train_item_user_matrix_fillna)
item_similarity_df = pd.DataFrame(item_similarity, index=train_item_user_matrix.index, columns=train_item_user_matrix.index)

print("Defining prediction function...")

# Define prediction function
def predict(user_id, item_id, k=20):
    # Return global average rating if the item is not in the similarity matrix
    if item_id not in item_similarity_df.index:
        return train_data['Rating'].mean()
    # Return item's average rating if the user is not in the training set
    if user_id not in train_item_user_matrix.columns:
        item_mean = train_item_user_matrix.loc[item_id].mean()
        if np.isnan(item_mean):
            return train_data['Rating'].mean()
        else:
            return item_mean
    # Get items similar to the target item
    sim_items = item_similarity_df[item_id].drop(item_id).dropna()
    sim_items = sim_items[sim_items > 0]
    sim_items = sim_items.sort_values(ascending=False)
    # Return item's average rating if no similar items are found
    if sim_items.empty:
        item_mean = train_item_user_matrix.loc[item_id].mean()
        if np.isnan(item_mean):
            return train_data['Rating'].mean()
        else:
            return item_mean
    # Select top k similar items
    top_k_items = sim_items.iloc[:k]
    # Get the user's ratings for these items
    top_k_ratings = train_item_user_matrix.loc[top_k_items.index, user_id]
    # Compute weighted average rating
    top_k_sim = top_k_items[top_k_ratings.notnull()]
    top_k_ratings = top_k_ratings[top_k_ratings.notnull()]
    if top_k_sim.sum() == 0:
        item_mean = train_item_user_matrix.loc[item_id].mean()
        if np.isnan(item_mean):
            return train_data['Rating'].mean()
        else:
            return item_mean
    else:
        return np.dot(top_k_sim, top_k_ratings) / top_k_sim.sum()

print("Predicting ratings for the test set...")

# Predict on the test set
test_data['Prediction'] = test_data.apply(lambda x: predict(x['UserID'], x['MovieID']), axis=1)

print("Predictions completed. Calculating evaluation metrics...")

# Calculate evaluation metrics

# MAE
mae = mean_absolute_error(test_data['Rating'], test_data['Prediction'])
print(f"MAE: {mae}")

# RMSE
rmse = np.sqrt(mean_squared_error(test_data['Rating'], test_data['Prediction']))
print(f"RMSE: {rmse}")

# Convert to binary classification (e.g., rating >= 4 is positive)
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

# Visualization - generate bar chart
metrics = {'MAE': mae, 'RMSE': rmse, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(10, 6))
plt.bar(metric_names, metric_values, color='skyblue')
plt.title('Item-CF Evaluation Metrics Results')
plt.ylabel('Value')
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
plt.show()

print("Visualization completed.")