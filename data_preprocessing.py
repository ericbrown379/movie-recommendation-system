import pandas as pd
import numpy as np


## 1. Load the dataset
## Adjust paths based on folder structure


## u.data: userID, movieID, rating, timestamp
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

# print("First 5 rows of the rating data:")
# print(ratings.head())

movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None, names=['movie_id', 'title', 'release_date', 'video_release', 'IMDb_URL', 
                            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

# print("First 5 rows of the movie data:")
# print(movies.head())

## 2. Take care of miisng values
##Drop movies without titles or release dates
movies.dropna(subset=["title", "release_date"], inplace=True)


# Check for missing values after dropping rows
# print("\nMissing values in 'title' and 'release_date' AFTER dropna:")
# print(movies[['title', 'release_date']].isnull().sum())

# # Print the shape (number of rows and columns) after dropping rows
# print("\nShape of the movies dataset AFTER dropna:")
# print(movies.shape)

## 3. Extract a Feature
##  Extract the release year from the title (if it's in the format "Movie Title (Year)")
## '\((\d{4})\)' is a regular expression (regex) pattern used to extract a 4-digit number (typically a year) that is enclosed in parentheses
movies["year"] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

##The ratings are normalized between 0 and 1 to make them suitable for machine learning models.
ratings['rating'] = ratings['rating'] / ratings['rating'].max()

## Merge the ratings with the movie information on 'movie_id'
merged_data = pd.merge(ratings, movies[['movie_id', "title", "year"]], on='movie_id')

## Split data into training and test sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Step 8: Display processed datasets
print("Training Dataset Sample:")
print(train_data.head())

print("Test Dataset Sample:")
print(test_data.head())




