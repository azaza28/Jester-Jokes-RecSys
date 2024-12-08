import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import polars as pl
import pickle
import os


class CollaborativeMemoryModel:
    def __init__(self, method="user"):
        """
        Memory-based collaborative filtering model.

        Args:
            method (str): 'user' for user-user filtering, 'item' for item-item filtering.
        """
        self.method = method
        self.similarity_matrix = None
        self.ratings_matrix = None
        self.user_ids = None
        self.item_ids = None

    def train(self, ratings: pl.DataFrame):
        """
        Train the model by computing similarity matrices.

        Args:
            ratings (pl.DataFrame): Polars DataFrame with columns ['userId', 'jokeId', 'rating'].
        """
        if ratings.is_empty():
            raise ValueError("❌ Ratings DataFrame is empty. Please provide a valid training dataset.")

        print("⚙️  Extracting user and item IDs...")
        self.user_ids = ratings.select("userId").unique().to_series().to_list()
        self.item_ids = ratings.select("jokeId").unique().to_series().to_list()

        print(f"📦 Total users: {len(self.user_ids)}, Total items: {len(self.item_ids)}")

        # Step 2: Convert userId and jokeId to categorical indices
        print("⚙️  Converting userId and jokeId to categorical indices...")
        ratings = ratings.with_columns([
            pl.col("userId").cast(pl.Utf8).cast(pl.Categorical).alias("user_idx"),
            pl.col("jokeId").cast(pl.Utf8).cast(pl.Categorical).alias("item_idx")
        ])

        user_index = ratings.get_column("user_idx").to_physical().to_numpy().flatten()
        item_index = ratings.get_column("item_idx").to_physical().to_numpy().flatten()
        rating_values = ratings.get_column("rating").to_numpy().flatten()

        print(f"🔢 Building sparse matrix with shape ({len(self.user_ids)}, {len(self.item_ids)})...")

        # Step 3: Create a sparse ratings matrix
        self.ratings_matrix = csr_matrix(
            (rating_values, (user_index, item_index)),
            shape=(len(self.user_ids), len(self.item_ids))
        )

        if self.ratings_matrix.nnz == 0:
            raise ValueError("❌ The ratings matrix is empty. Please check if the input data is correct.")

        print(f"✅ Ratings matrix created with {self.ratings_matrix.nnz} non-zero entries.")

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
        if self.ratings_matrix is None:
            raise ValueError("❌ Ratings matrix is not set. Please ensure the model is trained properly.")
        if user_id not in self.user_ids:
            raise ValueError(f"❌ User ID {user_id} not found in the user index.")

        user_idx = self.user_ids.index(user_id)
        scores = self.similarity_matrix[user_idx] @ self.ratings_matrix
        recommended_indices = np.argsort(scores)[::-1][:top_n]
        recommended_joke_ids = [self.item_ids[i] for i in recommended_indices]
        return recommended_joke_ids

    def save_model(self, filepath: str):
        """
        Save the entire model to a pickle file.

        Args:
            filepath (str): Path where to save the model file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"💾 Saving entire CollaborativeMemoryModel to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Collaborative Memory-Based model saved successfully at {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """
        Load the model from a pickle file.

        Args:
            filepath (str): Path to the saved model file.

        Returns:
            CollaborativeMemoryModel: Loaded model instance.
        """
        print(f"📦 Loading entire CollaborativeMemoryModel from {filepath}...")
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Collaborative Memory-Based model loaded successfully from {filepath}")
        return model