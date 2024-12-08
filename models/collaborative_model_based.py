import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse import save_npz, load_npz
from implicit.als import AlternatingLeastSquares
import os


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
        self.user_item_sparse = None  # Store the sparse matrix

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
        self.user_item_sparse = sp.csr_matrix((ratings, (user_index, item_index)), shape=(num_users, num_items))

        # Step 4: Train the ALS model
        print(f"📈 Training ALS model with {num_users} users and {num_items} items...")
        self.model.fit(self.user_item_sparse)
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

        # Get the specific row for this user as required by the ALS model
        user_items = self.user_item_sparse[user_idx]

        # Get the recommendations using ALS
        recommendations = self.model.recommend(user_idx, user_items, N=top_n)

        recommendations = list(zip(recommendations[0], recommendations[1]))

        # Extract the first two elements from the tuple (joke_id, score)
        recommended_jokes = [self.item_ids[joke_id] for joke_id, score in recommendations]

        return recommended_jokes

    def save_model(self, path="../models/collaborative_filtering/"):
        """
        💾 Save the ALS model factors, user-item sparse matrix, and metadata.

        Args:
            path (str): Directory path where to save the model.
        """
        os.makedirs(path, exist_ok=True)

        print(f"💾 Saving model to {path}...")

        # Save ALS factors
        np.savez(os.path.join(path, 'model_factors.npz'),
                 item_factors=self.model.item_factors,
                 user_factors=self.model.user_factors,
                 user_ids=self.user_ids,
                 item_ids=self.item_ids)

        # Save the sparse user-item interaction matrix
        save_npz(os.path.join(path, 'user_item_sparse.npz'), self.user_item_sparse)

        print(f"✅ Model saved successfully at {path}")

    @staticmethod
    def load_model(path="../models/collaborative_filtering/"):
        """
        📦 Load the ALS model factors and metadata from disk.

        Args:
            path (str): Path from where to load the model.

        Returns:
            CollaborativeModelBased: The loaded model instance.
        """
        print(f"📦 Loading Collaborative Filtering model from {path}...")

        # Load ALS factors
        data = np.load(os.path.join(path, 'model_factors.npz'), allow_pickle=True)

        # Instantiate a new CollaborativeModelBased instance
        model_instance = CollaborativeModelBased()
        model_instance.model.item_factors = data['item_factors']
        model_instance.model.user_factors = data['user_factors']
        model_instance.user_ids = data['user_ids'].tolist()
        model_instance.item_ids = data['item_ids'].tolist()

        # Load the sparse user-item interaction matrix
        model_instance.user_item_sparse = load_npz(os.path.join(path, 'user_item_sparse.npz'))
        print(f"📐 Loaded User-Item Matrix shape: {model_instance.user_item_sparse.shape}")

        print(f"✅ Model loaded from {path} with {len(model_instance.user_ids)} users and {len(model_instance.item_ids)} items.")
        return model_instance