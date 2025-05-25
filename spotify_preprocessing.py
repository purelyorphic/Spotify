# Decoding the Beat: A Data-Driven Insight into Spotify's Recommendation System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Drop unnecessary columns
df.drop(['track_id', 'track_name', 'artist_name'], axis=1, inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Encode categorical features
df = pd.get_dummies(df, columns=['genre'], drop_first=True)

# Handle outliers using IQR for 'tempo'
Q1 = df['tempo'].quantile(0.25)
Q3 = df['tempo'].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[(df['tempo'] >= Q1 - 1.5 * IQR) & (df['tempo'] <= Q3 + 1.5 * IQR)]

# Summary statistics
summary = filtered_df.describe()

# Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(filtered_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("heatmap.png")

# Visualize genre distribution
genre_columns = [col for col in filtered_df.columns if col.startswith('genre_')]
genre_sums = filtered_df[genre_columns].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
genre_sums.plot(kind='bar')
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("genre_distribution.png")

# Save summary statistics to CSV
summary.to_csv("summary_statistics.csv")

# Save cleaned data
filtered_df.to_csv("Cleaned_SpotifyData.csv", index=False)

print("EDA completed successfully. Outputs saved.")
