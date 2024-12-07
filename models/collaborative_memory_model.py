import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import polars as pl
import pickle


class CollaborativeMemoryModel:
    def __init__(self, method="user"):
        """
        📘 Memory-based collaborative filtering model.

        Args:
            method (str): 'user' for user-user filtering, 'item' for item-item filtering.
        """
        self.method = method
        self.similarity_matrix = None
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None

    def train(self, ratings):
        """
        🚀 Train the model by computing similarity matrices.

        Args:
            ratings (pl.DataFrame): Polars DataFrame with columns ['userId', 'jokeId', 'rating'].
        """
        print("🔹 Extracting user and item indices...")

        # Step 1: Extract user and item IDs as lists
        self.user_ids = ratings.select("userId").unique().to_series().to_list()
        self.item_ids = ratings.select("jokeId").unique().to_series().to_list()

        # Step 2: Convert userId and jokeId to categorical indices
        print("🔹 Converting userId and jokeId to categorical indices...")
        ratings = ratings.with_columns([
            pl.col("userId").cast(pl.Utf8).cast(pl.Categorical).alias("user_idx"),
            pl.col("jokeId").cast(pl.Utf8).cast(pl.Categorical).alias("item_idx")
        ])

        # Extract the integer representations of the categorical indices
        user_index = ratings.get_column("user_idx").to_physical().to_numpy().flatten()
        item_index = ratings.get_column("item_idx").to_physical().to_numpy().flatten()

        print("🔹 Creating the sparse ratings matrix...")

        # Step 3: Create a sparse ratings matrix
        self.ratings_matrix = csr_matrix(
            (
                ratings.select("rating").to_numpy().flatten(),
                (user_index, item_index)
            )
        )

        # Step 4: Calculate the similarity matrix (user-user or item-item)
        if self.method == "user":
            print("🔹 Calculating user-user similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.ratings_matrix)
        elif self.method == "item":
            print("🔹 Calculating item-item similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.ratings_matrix.T)
        else:
            raise ValueError("❌ Method must be 'user' or 'item'.")

        print(f"✅ Similarity matrix of shape {self.similarity_matrix.shape} calculated successfully.")

    def recommend(self, user_id, top_n=10):
        """
        Recommend jokes for a user based on collaborative filtering.

        Args:
            user_id (int): ID of the user to recommend jokes for.
            top_n (int): Number of recommendations.

        Returns:
            list: Recommended joke IDs.
        """
        if self.similarity_matrix is None:
            raise ValueError("The model is not trained yet. Please call train() first.")

        if user_id not in self.user_ids:
            raise ValueError(f"User ID {user_id} not found in the user index.")

        # Step 1: Find the index for the given user_id
        user_idx = self.user_ids.index(user_id)
        user_similarity_vector = self.similarity_matrix[user_idx].reshape(1, -1)  # Ensure it is 2D

        if self.ratings_matrix is None:
            raise ValueError("Ratings matrix is not set. Please ensure the model is trained properly.")

        if self.ratings_matrix.ndim != 2:
            raise ValueError(f"Expected self.ratings_matrix to be 2D, but got shape {self.ratings_matrix.shape}")

        if user_similarity_vector.ndim != 2:
            raise ValueError(f"Expected user similarity vector to be 2D, but got shape {user_similarity_vector.shape}")

        print(f"User idx: {user_idx}, User similarity vector shape: {user_similarity_vector.shape}")
        print(f"Ratings matrix shape: {self.ratings_matrix.shape}")

        # Step 2: Calculate scores using the similarity matrix and ratings matrix
        scores = user_similarity_vector @ self.ratings_matrix

        # Step 3: Get the top-N joke indices
        recommended_indices = np.argsort(scores.flatten())[::-1][:top_n]

        # Step 4: Convert the indices to joke IDs
        recommended_joke_ids = [self.item_ids[i] for i in recommended_indices]

        return recommended_joke_ids

    def save_model(self, filepath: str):
        """
        💾 Save the model to a file using pickle.

        Args:
            filepath (str): The path to save the model.
        """
        print(f"💾 Saving CollaborativeMemoryModel to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump({
                'similarity_matrix': self.similarity_matrix,
                'user_ids': self.user_ids,
                'item_ids': self.item_ids
            }, f)
        print(f"✅ Collaborative Memory-Based model saved successfully at {filepath}")

    def load_model(self, filepath: str):
        """
        📦 Load the model from a pickle file.

        Args:
            filepath (str): The path to the saved model file.
        """
        print(f"📦 Loading CollaborativeMemoryModel from {filepath}...")
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.similarity_matrix = model_data['similarity_matrix']
        self.user_ids = model_data['user_ids']
        self.item_ids = model_data['item_ids']
        print(f"✅ Collaborative Memory-Based model loaded successfully from {filepath}")