import polars as pl
import numpy as np
import os
import json
import pickle  # For saving the whole object
from sklearn.metrics.pairwise import cosine_similarity
from models.baseline_model import BaselineModel  # Import the BaselineModel


class ContentBasedModel:
    """
    🤖 Content-Based Recommender using precomputed embeddings.
    """

    def __init__(self):
        """
        📦 Initialize the Content-Based Recommender model.
        """
        self.embeddings = None  # Embeddings for all jokes
        self.similarity_matrix = None  # Cosine similarity matrix for all jokes
        self.joke_ids = None  # List of joke IDs corresponding to the embeddings
        self.baseline_model = None  # Placeholder for the baseline model (loaded dynamically)

    def train(self, embeddings, joke_ids):
        """
        🚀 Trains the content-based model using precomputed embeddings.

        Args:
            embeddings (np.ndarray): Precomputed embeddings for jokes (shape: [n_jokes, embedding_dim]).
            joke_ids (list): List of joke IDs corresponding to each embedding.
        """
        if embeddings.shape[0] != len(joke_ids):
            raise ValueError("❌ The number of embeddings must match the number of joke IDs.")

        self.embeddings = embeddings
        self.joke_ids = joke_ids

        print("🔍 Calculating cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        print(f"✅ Cosine similarity matrix shape: {self.similarity_matrix.shape}")

    def recommend(self, user_id, train_df=None, top_n=10):
        """
        🔮 Recommends similar jokes for the given user ID.
        If the user has no prior interactions, fallback to Baseline recommendations.

        Args:
            user_id (int): User ID for which the recommendation is being made.
            train_df (pl.DataFrame, optional): The train set containing 'userId' and 'jokeId'. Used to find user's prior interactions.
            top_n (int): Number of similar jokes to return.

        Returns:
            list: Top-N recommended joke IDs.
        """
        if self.similarity_matrix is None or self.joke_ids is None:
            raise ValueError("❌ Model has not been trained. Please train the model first.")

        if train_df is not None:
            # Get the user's past joke interactions from the test DataFrame
            user_interacted_jokes = train_df.filter(pl.col("userId") == user_id)["jokeId"].to_list()
        else:
            user_interacted_jokes = []

        if not user_interacted_jokes:
            print(f"⚠️ No prior interactions found for User ID {user_id}. Falling back to Baseline recommendations.")

            # Load the Baseline model if it is not loaded
            if self.baseline_model is None:
                try:
                    print(f"📦 Loading Baseline Model from ../models/baseline_model/...")
                    self.baseline_model = BaselineModel.load_model("../models/baseline_model/")
                except Exception as e:
                    print(f"❌ Failed to load Baseline Model: {e}")
                    print(f"🎉 Fallback to random joke recommendations instead.")
                    fallback_recommendations = np.random.choice(self.joke_ids, top_n, replace=False).tolist()
                    return fallback_recommendations

            # Recommend using the Baseline model
            print(f"🌟 Using Baseline Model to recommend for User ID {user_id}...")
            recommended_jokes = self.baseline_model.recommend(top_n=top_n)
            print(f"🎉 Baseline Recommendations for User ID {user_id}: {recommended_jokes}")
            return recommended_jokes

        # Use one of the user's interacted jokes as the "anchor" for recommendations
        random_joke_id = np.random.choice(user_interacted_jokes)

        # If the selected joke is not part of the current joke pool, fallback to Baseline
        if random_joke_id not in self.joke_ids:
            print(
                f"⚠️ Joke ID {random_joke_id} is not in the model. Falling back to Baseline recommendations for User ID {user_id}.")
            return self.baseline_model.recommend(top_n=top_n)

        # Generate recommendations using the content-based method
        idx = self.joke_ids.index(random_joke_id)
        similarity_scores = self.similarity_matrix[idx]
        top_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]
        recommended_jokes = [self.joke_ids[i] for i in top_indices]

        return recommended_jokes

    def save_model(self, filepath):
        """
        💾 Save the entire Content-Based model object as a pickle file.

        Args:
            filepath (str): File path where the model should be saved (should end with .pkl).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Model saved successfully to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        📦 Load the entire Content-Based model object from a pickle file.

        Args:
            filepath (str): File path from where the model should be loaded.

        Returns:
            ContentBasedModel: The loaded Content-Based model instance.
        """
        if not os.path.exists(filepath):
            raise ValueError(f"❌ File not found: {filepath}")

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        print(f"📦 Model loaded successfully from {filepath}")
        return model