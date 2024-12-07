import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class CollaborativeModelBased:
    """
    🤖 Model-based collaborative filtering using Implicit ALS.
    """

    def __init__(self, factors=64, iterations=20, regularization=0.1):
        """
        📦 Initialize the ALS model.

        Args:
            factors (int): Number of latent factors.
            iterations (int): Number of ALS iterations.
            regularization (float): Regularization coefficient.
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            use_gpu=False
        )
        self.user_ids = None
        self.item_ids = None

    def train(self, train_data: pl.DataFrame):
        """
        🚀 Trains a collaborative filtering model using ALS.

        Args:
            train_data (pl.DataFrame): Polars DataFrame with ['userId', 'jokeId', 'rating'].
        """
        print("🔹 Starting training for Collaborative Filtering Model...")

        # Step 1: Map userId and jokeId to dense indices
        train_data = train_data.with_columns([
            pl.col("userId").cast(pl.Utf8).cast(pl.Categorical).alias("user_idx"),
            pl.col("jokeId").cast(pl.Utf8).cast(pl.Categorical).alias("item_idx")
        ])

        self.user_ids = train_data.get_column("userId").unique().to_list()
        self.item_ids = train_data.get_column("jokeId").unique().to_list()

        # Step 2: Extract user_idx, item_idx, and ratings
        user_index = train_data.get_column("user_idx").to_physical().to_numpy().flatten()
        item_index = train_data.get_column("item_idx").to_physical().to_numpy().flatten()
        ratings = train_data.get_column("rating").to_numpy().astype(np.float32)

        # Step 3: Create the sparse user-item interaction matrix
        num_users = len(self.user_ids)
        num_items = len(self.item_ids)
        user_item_sparse = sp.csr_matrix((ratings, (user_index, item_index)), shape=(num_users, num_items))

        # Step 4: Train the ALS model
        print(f"📈 Training ALS model with {num_users} users and {num_items} items...")
        self.model.fit(user_item_sparse)
        print("✅ Training complete!")

    def recommend(self, user_id: int, top_n=10):
        """
        🔮 Recommend top-N jokes for a given user.

        Args:
            user_id (int): User ID for which to recommend jokes.
            top_n (int): Number of recommendations.

        Returns:
            list: List of top-N recommended joke IDs.
        """
        if user_id not in self.user_ids:
            raise ValueError(f"❌ User ID {user_id} not found.")

        user_idx = self.user_ids.index(user_id)
        recommendations = self.model.recommend(user_idx, self.model.item_factors, N=top_n)

        recommended_jokes = [self.item_ids[joke_id] for joke_id, score in recommendations]
        print(f"🔮 Recommendations for user {user_id}: {recommended_jokes}")

        return recommended_jokes

    def save_model(self, path="../models/collaborative_filtering.npz"):
        """
        💾 Save the ALS model factors and metadata to disk.

        Args:
            path (str): Path where to save the model.
        """
        np.savez(path,
                 item_factors=self.model.item_factors,
                 user_factors=self.model.user_factors,
                 user_ids=self.user_ids,
                 item_ids=self.item_ids)
        print(f"✅ Model saved at {path}")

    def load_model(self, path="../models/collaborative_filtering.npz"):
        """
        📦 Load the ALS model factors and metadata from disk.

        Args:
            path (str): Path from where to load the model.
        """
        print(f"📦 Loading Collaborative Filtering model from {path}...")
        data = np.load(path, allow_pickle=True)

        self.model.item_factors = data['item_factors']
        self.model.user_factors = data['user_factors']
        self.user_ids = data['user_ids'].tolist()
        self.item_ids = data['item_ids'].tolist()

        print(f"✅ Model loaded from {path} with {len(self.user_ids)} users and {len(self.item_ids)} items.")