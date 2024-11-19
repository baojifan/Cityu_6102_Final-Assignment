import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

os.chdir('/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset')

# Read data
data = pd.read_csv('ml-1m/processed_merged_data.csv')

# Extract genre columns
genre_columns = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
                 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                 'Sci-Fi', 'Thriller', 'War', 'Western']

# Calculate the total count of each genre
genre_counts = data[genre_columns].sum()

# Convert genres and their frequencies into a dictionary
genre_dict = genre_counts.to_dict()

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(genre_dict)

# Display word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axis
plt.title('Movie Genres Word Cloud', fontsize=20)
plt.show()