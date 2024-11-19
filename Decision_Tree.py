import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Import functions from split_utils.py
from split_utils import min_rating_filter_pandas, split_pandas_data_with_ratios

# Load data
print("Loading data...")
movies = pd.read_csv('movies.dat', sep='::', header=None, names=['MovieID', 'Title', 'Genres'],
                     engine='python', encoding='ISO-8859-1')
users = pd.read_csv('users.dat', sep='::', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                    engine='python', encoding='UTF-8')
ratings = pd.read_csv('ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                      engine='python', encoding='UTF-8')
print("Data loading completed.")

# Data preprocessing
print("Data preprocessing...")

# Filter users and movies with minimum number of ratings
print("Filtering users and movies with minimum number of ratings...")
# Filter users with fewer than 5 ratings
ratings_filtered = min_rating_filter_pandas(ratings, min_rating=5, filter_by='user',
                                           col_user='UserID', col_item='MovieID')
# Filter movies with fewer than 5 ratings
ratings_filtered = min_rating_filter_pandas(ratings_filtered, min_rating=5, filter_by='item',
                                           col_user='UserID', col_item='MovieID')

# Merge datasets
data = pd.merge(pd.merge(ratings_filtered, users), movies)

# Process Gender feature
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})

# Map Age to categories and encode
age_map = {1: 'Under 18', 18: '18-24', 25: '25-34',
           35: '35-44', 45: '45-49', 50: '50-55', 56: '56+'}
data['Age'] = data['Age'].map(age_map)
age_dummies = pd.get_dummies(data['Age'], prefix='Age')
data = pd.concat([data, age_dummies], axis=1)
data.drop('Age', axis=1, inplace=True)

# One-hot encode Occupation
occupation_dummies = pd.get_dummies(data['Occupation'], prefix='Occ')
data = pd.concat([data, occupation_dummies], axis=1)
data.drop('Occupation', axis=1, inplace=True)

# Process Genres feature via one-hot encoding
genres_dummies = data['Genres'].str.get_dummies(sep='|')
data = pd.concat([data, genres_dummies], axis=1)
data.drop('Genres', axis=1, inplace=True)

# Remove unnecessary columns
data.drop(['Title', 'Zip-code', 'Timestamp'], axis=1, inplace=True)

print("Data preprocessing completed.")

# Build features and labels
print("Building features and labels...")
X = data.drop('Rating', axis=1)
y = data['Rating']

# Split data into training and test sets
print("Splitting data into training and test sets...")

# Split dataset using split_pandas_data_with_ratios
train_data, test_data = split_pandas_data_with_ratios(data, ratios=[0.8, 0.2], seed=42, shuffle=True)

# Since split_pandas_data_with_ratios returns lists, process accordingly
train_data = train_data.copy()
test_data = test_data.copy()

# Remove 'split_index' column if it exists
if 'split_index' in train_data.columns:
    train_data = train_data.drop(columns=['split_index'])
if 'split_index' in test_data.columns:
    test_data = test_data.drop(columns=['split_index'])

# Extract X and y from train_data and test_data
X_train = train_data.drop('Rating', axis=1)
y_train = train_data['Rating']
X_test = test_data.drop('Rating', axis=1)
y_test = test_data['Rating']

# Train Decision Tree model
print("Training Decision Tree model...")
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
print("Predicting and evaluating...")
y_pred = model.predict(X_test)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Convert ratings to like (1) and dislike (0)
threshold = 4.5  # Ratings greater than or equal to 4 are considered liked
y_test_class = (y_test >= threshold).astype(int)
y_pred_class = (y_pred >= threshold).astype(int)

# Calculate Precision, Recall, F1-Score
precision = precision_score(y_test_class, y_pred_class, zero_division=0)
recall = recall_score(y_test_class, y_pred_class, zero_division=0)
f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Visualize the results
print("Visualizing the results...")
# Create a dictionary of evaluation metrics
metrics = {
    'MAE': mae,
    'RMSE': rmse,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}

# Plot bar chart
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values, color='skyblue')
plt.title('Decision Tree Evaluation Metrics')
plt.ylabel('Score')

# Display values on each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01,
             f"{yval:.4f}", ha='center', va='bottom')

plt.show()