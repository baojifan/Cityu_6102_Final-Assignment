import pandas as pd
'''
# File paths
filename1 = r'/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset/ml-1m/users.dat'
filename2 = r'/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset/ml-1m/ratings.dat'
filename3 = r'/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset/ml-1m/movies.dat'

# Set max display rows
pd.options.display.max_rows = 10

# Read user data
uname = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(filename1, sep='::', header=None, names=uname, engine='python', encoding='utf-8')

# Read rating data
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(filename2, header=None, sep='::', names=rnames, engine='python', encoding='utf-8')

# Read movie data with the correct encoding format (assumed to be ISO-8859-1)
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(filename3, header=None, sep='::', names=mnames, engine='python', encoding='ISO-8859-1')

# Merge the three tables using the merge function
data = pd.merge(pd.merge(ratings, users), movies)

# Save the merged data as a CSV file
output_file = '/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset/ml-1m/merged_data.csv'
data.to_csv(output_file, index=False, encoding='utf-8')

print(f"Merged data has been saved to: {output_file}")  
'''

# Define movie genres list
genres_list = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]

# Read merged_data.csv file
input_file = '/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset/ml-1m/merged_data.csv'
data = pd.read_csv(input_file)

# Perform one-hot encoding on the genres column, where each genre gets its own column (1 if the movie belongs to that genre, 0 otherwise)
genres_dummies = data['genres'].str.get_dummies(sep='|')

# Ensure the generated one-hot encoded columns correspond to the given genres list
for genre in genres_list:
    if genre not in genres_dummies.columns:
        genres_dummies[genre] = 0  # If a genre column is missing, manually add a column with all 0s

# Rearrange the columns to match the given order
genres_dummies = genres_dummies[genres_list]

# Merge the one-hot encoded genres with the original data and remove the original genres column
data = pd.concat([data.drop('genres', axis=1), genres_dummies], axis=1)

# Save the processed data to a new CSV file
output_file = '/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset/ml-1m/processed_merged_data.csv'
data.to_csv(output_file, index=False)

print(f"Processed data has been saved to: {output_file}")