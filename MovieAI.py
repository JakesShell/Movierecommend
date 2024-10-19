import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the MovieLens dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Display the first few rows of the ratings
print(ratings.head())
print(movies.head())

# Prepare the data for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build the collaborative filtering model using SVD
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Calculate and print RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

# Function to get movie recommendations
def get_recommendations(user_id, num_recommendations=5):
    movie_ids = ratings['movieId'].unique()
    user_ratings = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_movie_ids = set(movie_ids) - set(user_ratings)
    
    # Predict ratings for all movies the user hasn't seen
    recommendations = []
    for movie_id in all_movie_ids:
        pred_rating = model.predict(user_id, movie_id).est
        recommendations.append((movie_id, pred_rating))
    
    # Sort recommendations by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    top_recommendations = recommendations[:num_recommendations]
    
    # Return movie titles along with predicted ratings
    recommended_movies = []
    for movie_id, rating in top_recommendations:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        recommended_movies.append((title, rating))
    
    return recommended_movies

# Example usage
user_id = 1  # Change this to any user ID in the dataset
recommendations = get_recommendations(user_id)
print(f"Recommendations for User {user_id}:")
for title, rating in recommendations:
    print(f"{title} - Predicted Rating: {rating:.2f}")
