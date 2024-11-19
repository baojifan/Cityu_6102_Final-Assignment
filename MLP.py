# Import necessary libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ignore ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
print("Data loaded successfully.")

# Data preprocessing
print("Preprocessing data...")

# Filter users and movies with few ratings
print("Filtering users and movies with few ratings...")
ratings_filtered = min_rating_filter_pandas(ratings, min_rating=5, filter_by='user',
                                            col_user='UserID', col_item='MovieID')
ratings_filtered = min_rating_filter_pandas(ratings_filtered, min_rating=5, filter_by='item',
                                            col_user='UserID', col_item='MovieID')

# Merge datasets
data = pd.merge(pd.merge(ratings_filtered, users), movies)

# Process Gender feature
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})

# Map Age to categories and one-hot encode
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

# Process Genres feature and one-hot encode
genres_dummies = data['Genres'].str.get_dummies(sep='|')
data = pd.concat([data, genres_dummies], axis=1)
data.drop('Genres', axis=1, inplace=True)

# Drop unnecessary columns
data.drop(['Title', 'Zip-code', 'Timestamp'], axis=1, inplace=True)

print("Data preprocessing completed.")

# Build features and labels
print("Building features and labels...")
X = data.drop('Rating', axis=1)
y = data['Rating']

# Standardize numerical features
print("Standardizing numerical features...")
numeric_features = ['UserID', 'MovieID']
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split data into training, validation, and test sets
print("Splitting data into training, validation, and test sets...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1, random_state=42)

# Train MLP model and record Precision and Recall
print("Training MLP model and recording Precision and Recall...")
max_epochs = 100
iterations_per_epoch = 5
precision_list = []
recall_list = []

model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                     random_state=42, warm_start=True, max_iter=iterations_per_epoch)

for epoch in range(max_epochs):
    print(f"Epoch {epoch+1}/{max_epochs}")
    model.max_iter += iterations_per_epoch
    model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val)

    # Convert ratings to binary (like or dislike)
    threshold = 4
    y_val_class = (y_val >= threshold).astype(int)
    y_val_pred_class = (y_val_pred >= threshold).astype(int)

    # Compute Precision and Recall
    precision = precision_score(y_val_class, y_val_pred_class, zero_division=0)
    recall = recall_score(y_val_class, y_val_pred_class, zero_division=0)

    precision_list.append(precision)
    recall_list.append(recall)

# Evaluate the model on the test set
print("Predicting and evaluating on the test set...")
y_pred = model.predict(X_test)

# Compute MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Convert ratings to binary (like or dislike)
y_test_class = (y_test >= threshold).astype(int)
y_pred_class = (y_pred >= threshold).astype(int)

# Compute Precision, Recall, and F1-Score
precision = precision_score(y_test_class, y_pred_class, zero_division=0)
recall = recall_score(y_test_class, y_pred_class, zero_division=0)
f1 = f1_score(y_test_class, y_pred_class, zero_division=0)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Visualize results
print("Visualizing results...")

# Plot Precision and Recall curves
'''
epochs = range(1, max_epochs + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, precision_list, 'b', label='Precision')
plt.plot(epochs, recall_list, 'r', label='Recall')
plt.title('Precision and Recall over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()
'''

# Plot evaluation metrics bar chart
metrics = {'MAE': mae, 'RMSE': rmse,
           'Precision': precision, 'Recall': recall, 'F1-Score': f1}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values, color='skyblue')
plt.title('MLP Evaluation Metrics')
plt.ylabel('Score')

# Display values on each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0,
             yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

plt.show()