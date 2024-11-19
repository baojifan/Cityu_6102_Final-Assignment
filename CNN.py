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
print("Data loaded.")

# Data preprocessing
print("Preprocessing data...")

# Filter users and movies with fewer ratings using the function from split_utils.py
print("Filtering users and movies with fewer ratings...")
ratings_filtered = min_rating_filter_pandas(ratings, min_rating=5, filter_by='user',
                                            col_user='UserID', col_item='MovieID')
ratings_filtered = min_rating_filter_pandas(ratings_filtered, min_rating=5, filter_by='item',
                                            col_user='UserID', col_item='MovieID')

# Merge datasets
data = pd.merge(pd.merge(ratings_filtered, users), movies)

# Process timestamp data and sort by time
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
data = data.sort_values(by=['UserID', 'Timestamp'])

# Generate movie watching sequences for each user
print("Generating movie watching sequences for each user...")
user_sequences = data.groupby('UserID')['MovieID'].apply(list).reset_index()
user_ratings = data.groupby('UserID')['Rating'].apply(list).reset_index()

# Map movie IDs to continuous indices
print("Mapping movie IDs to continuous indices...")
movie_id_mapping = {id: idx for idx, id in enumerate(data['MovieID'].unique())}
reverse_movie_id_mapping = {idx: id for id, idx in movie_id_mapping.items()}
data['MovieID_idx'] = data['MovieID'].map(movie_id_mapping)

# Prepare sequence data
print("Preparing sequence data...")
sequences = []
targets = []
sequence_length = 5  # Define sequence length

for user_id in user_sequences['UserID']:
    movie_ids = data[data['UserID'] == user_id]['MovieID_idx'].tolist()
    ratings_list = data[data['UserID'] == user_id]['Rating'].tolist()
    for i in range(len(movie_ids) - sequence_length):
        sequences.append(movie_ids[i:i + sequence_length])
        targets.append(ratings_list[i + sequence_length])

# Convert to NumPy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Split into training and test sets
print("Splitting training and test sets...")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    sequences, targets, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create data loaders
batch_size = 512
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Build CNN model
print("Building CNN model...")
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, dropout=0.5):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        x = self.embedding(x)  # x: [batch_size, seq_len, embedding_dim]
        x = x.unsqueeze(1)  # x: [batch_size, 1, seq_len, embedding_dim]
        conved = [nn.functional.relu(conv(x).squeeze(3)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        out = self.fc(cat)
        return out.squeeze()


# Define model hyperparameters
vocab_size = len(movie_id_mapping)
embedding_dim = 32
num_filters = 100
filter_sizes = [2, 3, 4]
dropout = 0.5

model = CNNModel(vocab_size, embedding_dim, num_filters, filter_sizes, dropout)

# Define loss function and optimizer
criterion = nn.MSELoss()
learning_rate = 0.0001
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train the model and record loss
print("Training the model and recording loss...")
epochs = 100
threshold = 4  # Ratings greater than or equal to 4 are considered liked

training_losses = []
validation_losses = []
best_val_loss = float('inf')
patience = 3  # Tolerance
trigger_times = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        epoch_loss += loss.item()
    avg_training_loss = epoch_loss / len(train_loader)
    training_losses.append(avg_training_loss)

    # Validation
    model.eval()
    val_loss = 0
    y_pred_list = []
    y_test_list = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            y_pred_list.extend(outputs.numpy())
            y_test_list.extend(targets.numpy())
    avg_val_loss = val_loss / len(test_loader)
    validation_losses.append(avg_val_loss)

    print(f"Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trigger_times = 0
        # torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
model.eval()
y_pred_list = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        y_pred_list.extend(outputs.numpy())

y_pred_array = np.array(y_pred_list)
y_test_array = y_test_tensor.numpy()
y_pred_classes = (y_pred_array >= threshold).astype(int)
y_test_classes = (y_test_array >= threshold).astype(int)

from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test_array, y_pred_array)
rmse = np.sqrt(mean_squared_error(y_test_array, y_pred_array))

# Calculate Precision, Recall, F1-Score
precision = precision_score(y_test_classes, y_pred_classes, zero_division=0)
recall = recall_score(y_test_classes, y_pred_classes, zero_division=0)
f1 = f1_score(y_test_classes, y_pred_classes, zero_division=0)

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

# Display specific values on each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0,
             yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')
plt.show()