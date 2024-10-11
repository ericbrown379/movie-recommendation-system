import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load train_data from CSV
print("Loading train_data from 'train_data.csv'...")
train_data = pd.read_csv('train_data.csv')
print("train_data loaded successfully!")
print(train_data.head())  # Print first few rows of train_data to confirm

# Load the movies dataset
print("Loading movies dataset from 'ml-100k/u.item'...")
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None, names=[
    'movie_id', 'title', 'release_date', 'video_release', 'IMDb_URL', 'unknown', 
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
])
print("Movies dataset loaded successfully!")
print(movies.head())  # Print first few rows of movies dataset to confirm

# Step 1: Plot the Distribution of Ratings
print("Creating a count plot for 'rating' column...")
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=train_data)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
print("Rating distribution plot displayed.")

# Step 2: Plot Average Ratings per Movie
print("Calculating average ratings per movie...")
movie_ratings = train_data.groupby('title')['rating'].mean().sort_values(ascending=False)
print("Top 10 movies by average rating:")
print(movie_ratings.head(10))  # Print top 10 movies
plt.figure(figsize=(10, 8))
movie_ratings.head(10).plot(kind='barh', color='skyblue')
plt.title('Top 10 Movies by Average Rating')
plt.xlabel('Average Rating')
plt.show()

# Step 3: Visualize Most Popular Genres
print("Calculating popularity of each genre...")
genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

genre_popularity = movies[genre_columns].sum().sort_values(ascending=False)
print("Most popular genres:")
print(genre_popularity.head())  # Print most popular genres
plt.figure(figsize=(10, 6))
genre_popularity.plot(kind='bar', color='coral')
plt.title('Most Popular Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.show()

# Step 4: Heatmap of Correlations
print("Generating correlation matrix heatmap...")
corr_matrix = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
print("Correlation heatmap displayed.")