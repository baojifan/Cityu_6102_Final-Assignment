import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#Load data
users_columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
users = pd.read_csv('users.dat', sep='::', names=users_columns, engine='python', encoding='UTF-8')

movies_columns = ['movie_id', 'title', 'genres']
movies = pd.read_csv('movies.dat', sep='::', names=movies_columns, engine='python', encoding='ISO-8859-1')

ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ratings.dat', sep='::', names=ratings_columns, engine='python', encoding='UTF-8')

# Data preprocessing
train_data, test_data = train_test_split(ratings, test_size=0.3, random_state=42)

# Implement random recommender model
class RandomRecommender:
    def __init__(self, min_rating=1, max_rating=5, recommend_ratio=1.0):
        self.min_rating = min_rating
        self.max_rating = max_rating
        # recommend_ratio indicates the proportion of items to recommend, e.g., 0.5 means randomly recommending 50% of items
        self.recommend_ratio = recommend_ratio

    def fit(self, train_data):
        # No training process needed here, just using the rating range in the dataset
        pass

    def predict(self, user_id, movie_id):
        # Randomly decide whether to recommend an item, independent of ratings
        recommend = np.random.rand() < self.recommend_ratio
        if recommend:
            return np.random.uniform(self.min_rating, self.max_rating)
        else:
            return 0  # If not recommended, the item's rating is 0

    def test(self, test_data):
        predictions = []
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            true_rating = row['rating']

            # Randomly decide whether to recommend
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((user_id, movie_id, true_rating, pred_rating))
        return predictions


# Create an instance of the random recommender model, with the recommendation ratio set to 50%
random_rec = RandomRecommender(recommend_ratio=1.0)

# Train the model (actually no training is needed here)
random_rec.fit(train_data)

# Evaluate the model
# Predict on the test set
predictions = random_rec.test(test_data)

# Extract true ratings and predicted ratings
true_ratings = [true for (_, _, true, _) in predictions]
pred_ratings = [pred for (_, _, _, pred) in predictions]

# Calculate MAE and RMSE
mae = mean_absolute_error(true_ratings, pred_ratings)
rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))

# Set the recommendation threshold (e.g., items with a rating greater than 4.0 are considered recommended)
threshold = 4.5

# Binarize true ratings and predicted ratings
true_ratings_binary = np.array(true_ratings) >= threshold
pred_ratings_binary = np.array(pred_ratings) >= threshold

print(f"Number of items actually liked: {np.sum(true_ratings_binary)}")
print(f"Number of items recommended by the model: {np.sum(pred_ratings_binary)}")

# Calculate Precision, Recall, and F1-score, zero_division=0 avoids overestimating Precision
precision = precision_score(true_ratings_binary, pred_ratings_binary, zero_division=0)
recall = recall_score(true_ratings_binary, pred_ratings_binary, zero_division=0)
f1 = f1_score(true_ratings_binary, pred_ratings_binary, zero_division=0)

# Print evaluation results
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Visualize evaluation metrics
metrics = {'MAE': mae, 'RMSE': rmse, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.figure(figsize=(8, 6))
plt.bar(metric_names, metric_values, color=['skyblue'])
plt.title('Evaluation Metrics for Random Recommender')
plt.ylabel('Score')
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
plt.show()