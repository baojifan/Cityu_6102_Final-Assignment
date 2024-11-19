# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Import functions from split_utils.py
from split_utils import min_rating_filter_pandas

# Load data
print("Loading data...")
movies = pd.read_csv('movies.dat', sep='::', header=None,
                     names=['MovieID', 'Title', 'Genres'],
                     engine='python', encoding='ISO-8859-1')
users = pd.read_csv('users.dat', sep='::', header=None,
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                    engine='python', encoding='UTF-8')
ratings = pd.read_csv('ratings.dat', sep='::', header=None,
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                      engine='python', encoding='UTF-8')
print("Data loaded successfully.")

# Data preprocessing
print("Preprocessing data...")

# Filter users and movies with few ratings
ratings_filtered = min_rating_filter_pandas(ratings, min_rating=5, filter_by='user',
                                            col_user='UserID', col_item='MovieID')
ratings_filtered = min_rating_filter_pandas(ratings_filtered, min_rating=5, filter_by='item',
                                            col_user='UserID', col_item='MovieID')

# Merge datasets
data = pd.merge(pd.merge(ratings_filtered, users), movies)

# Convert ratings to binary labels (like or dislike)
threshold = 4.5
data['Label'] = (data['Rating'] >= threshold).astype(int)

# Process user features
print("Processing user features...")
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})

# Map age to categories
age_map = {1: 'Under 18', 18: '18-24', 25: '25-34',
           35: '35-44', 45: '45-49', 50: '50-55', 56: '56+'}
data['Age'] = data['Age'].map(age_map)

# One-hot encode age and occupation
age_dummies = pd.get_dummies(data['Age'], prefix='Age')
occupation_dummies = pd.get_dummies(data['Occupation'], prefix='Occ')
data = pd.concat([data, age_dummies, occupation_dummies], axis=1)

# Process movie features
print("Processing movie features...")
genre_dummies = data['Genres'].str.get_dummies(sep='|')
data = pd.concat([data, genre_dummies], axis=1)

# Drop unnecessary columns
data.drop(['UserID', 'MovieID', 'Title', 'Genres', 'Zip-code', 'Timestamp', 'Age', 'Occupation', 'Rating'], axis=1,
          inplace=True)

# Features and labels
X = data.drop('Label', axis=1)
y = data['Label']

# Standardize numerical features
from sklearn.preprocessing import StandardScaler

numeric_features = ['Gender']
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split the dataset into training and testing sets
print("Splitting data into training and test sets...")
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Further split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1, random_state=42)

# Train logistic regression model and record loss
print("Training logistic regression model and recording loss...")
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

epochs = 20
training_losses = []
validation_losses = []

model = SGDClassifier(loss='log_loss', max_iter=1, learning_rate='optimal', tol=None, random_state=42, warm_start=True)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.fit(X_train, y_train)

    # Compute training loss
    y_train_proba = model.predict_proba(X_train)
    train_loss = log_loss(y_train, y_train_proba)
    training_losses.append(train_loss)

    # Compute validation loss
    y_val_proba = model.predict_proba(X_val)
    val_loss = log_loss(y_val, y_val_proba)
    validation_losses.append(val_loss)

    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Evaluate the model on the test set
print("Evaluating model on the test set...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mae = mean_absolute_error(y_test, y_pred_proba)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Visualize results
print("Visualizing results...")

# Plot training and validation loss curves
epochs_range = range(1, len(training_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, training_losses, 'b', label='Training Loss')
plt.plot(epochs_range, validation_losses, 'r', label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot evaluation metrics bar chart
metrics = {'MAE': mae, 'RMSE': rmse,
           'Precision': precision, 'Recall': recall, 'F1-Score': f1}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values, color='skyblue')
plt.title('Evaluation Metrics')
plt.ylabel('Score')

# Display values on each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0,
             yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')
plt.show()