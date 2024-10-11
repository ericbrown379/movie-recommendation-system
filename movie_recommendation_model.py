import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Step 1: Load the preprocessed training and testing data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Step 2: Define the number of unique users and movies
num_users = train_data['user_id'].nunique()
num_movies = train_data['movie_id'].nunique()

# Step 3: Define the embedding dimension (size of embedding vectors)
embedding_size = 50  # This is a tunable hyperparameter

# Step 4: Define the Neural Network Model with Embedding Layers for Collaborative Filtering
class RecommenderNet(Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        # User embedding layer
        self.user_embedding = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )

        # Movie embedding layer
        self.movie_embedding = layers.Embedding(
            input_dim=num_movies,
            output_dim=embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )

        # User bias and movie bias
        self.user_bias = layers.Embedding(input_dim=num_users, output_dim=1)
        self.movie_bias = layers.Embedding(input_dim=num_movies, output_dim=1)

    def call(self, inputs):
        user_id, movie_id = inputs

        # Flatten the input IDs
        user_id = tf.reshape(user_id, (-1,))
        movie_id = tf.reshape(movie_id, (-1,))

        # Get embeddings
        user_embedding = self.user_embedding(user_id)
        movie_embedding = self.movie_embedding(movie_id)

        # Get biases
        user_bias = self.user_bias(user_id)
        movie_bias = self.movie_bias(movie_id)

        # Compute dot product of user and movie embeddings
        dot_product = tf.reduce_sum(user_embedding * movie_embedding, axis=1)

        # Add biases
        x = dot_product + tf.reshape(user_bias, (-1,)) + tf.reshape(movie_bias, (-1,))

        return x

# Step 5: Prepare the model
model = RecommenderNet(num_users=num_users, num_movies=num_movies, embedding_size=embedding_size)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Step 6: Prepare the data for TensorFlow
# Include ratings in the dataset
train_data_tf = tf.data.Dataset.from_tensor_slices((
    tf.cast(train_data['user_id'].values, tf.int32),
    tf.cast(train_data['movie_id'].values, tf.int32),
    tf.cast(train_data['rating'].values, tf.float32)
))

train_data_tf = train_data_tf.map(
    lambda user_id, movie_id, rating: (
        (tf.expand_dims(user_id, -1), tf.expand_dims(movie_id, -1)),
        rating
    )
).shuffle(10000).batch(256)

test_data_tf = tf.data.Dataset.from_tensor_slices((
    tf.cast(test_data['user_id'].values, tf.int32),
    tf.cast(test_data['movie_id'].values, tf.int32),
    tf.cast(test_data['rating'].values, tf.float32)
))

test_data_tf = test_data_tf.map(
    lambda user_id, movie_id, rating: (
        (tf.expand_dims(user_id, -1), tf.expand_dims(movie_id, -1)),
        rating
    )
).batch(256)

# Step 7: Train the Model
history = model.fit(train_data_tf, epochs=10, validation_data=test_data_tf)

# Step 8: Evaluate the Model
model.evaluate(test_data_tf)

# Step 9: Save the Model (Updated)
model.save('movie_recommendation_model.keras')

# Step 10: Visualize Training History
# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot mean absolute error
##Hey bracnh practice
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()