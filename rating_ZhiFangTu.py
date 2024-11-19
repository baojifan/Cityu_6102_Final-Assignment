import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir('/Users/bao_jifan/Documents/coding/python_code/cityu_6102/ml_1m/dataset')

# Read the filtered movie ratings dataset
df = pd.read_csv('ml-1m/processed_merged_data.csv')

# Set plot style
sns.set(style="white")

# Create a layout with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Plot histogram of ratings - distribution of ratings
sns.histplot(df['rating'], bins=9, kde=False, color='green', ax=axes[0])

# Set title and labels
axes[0].set_title('Distribution of Ratings', fontsize=14)
axes[0].set_xlabel('Rating', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)

# 2. Plot box plot of ratings - ratings on the y-axis
sns.boxplot(y=df['rating'], ax=axes[1], color='lightblue')

# Set title and labels
axes[1].set_title('Box Plot of Ratings', fontsize=14)
axes[1].set_ylabel('Rating', fontsize=12)

# Automatically adjust the spacing between subplots
plt.tight_layout()

# Show the combined plots
plt.show()