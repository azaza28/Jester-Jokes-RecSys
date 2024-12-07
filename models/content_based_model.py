import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity


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

    def recommend(self, joke_id, top_n=10):
        """
        Recommends similar jokes to the given joke ID.

        Args:
            joke_id (int): ID of the joke to recommend similar jokes for.
            top_n (int): Number of similar jokes to return.

        Returns:
            list: Top-N recommended joke IDs.
        """
        if self.similarity_matrix is None or self.joke_ids is None:
            raise ValueError("❌ Model has not been trained. Please train the model first.")

        # 🔥 Check if the joke ID exists in the list of joke IDs
        if joke_id not in self.joke_ids:
            print(f"❌ Joke ID {joke_id} not found in the dataset.")

            # Fallback to recommend a random joke
            random_recommendation = np.random.choice(self.joke_ids, size=top_n, replace=False).tolist()
            print(f"🎉 Fallback Recommendation: {random_recommendation}")
            return random_recommendation

    def recommend(self, joke_id, top_n=10):
        """
        Recommends similar jokes to the given joke ID.

        Args:
            joke_id (int): ID of the joke to recommend similar jokes for.
            top_n (int): Number of similar jokes to return.

        Returns:
            list: Top-N recommended joke IDs (or an empty list if something goes wrong).
        """
        if self.similarity_matrix is None or self.joke_ids is None:
            print("❌ Model has not been trained. Please train the model first.")
            random_recommendation = np.random.choice(self.joke_ids, size=top_n, replace=False).tolist()
            print(f"🎉 Fallback Recommendation: {random_recommendation}")
            return random_recommendation

        if joke_id not in self.joke_ids:
            print(f"❌ Joke ID {joke_id} not found in the dataset.")
            random_recommendation = np.random.choice(self.joke_ids, size=top_n, replace=False).tolist()
            print(f"🎉 Fallback Recommendation: {random_recommendation}")
            return random_recommendation

        idx = self.joke_ids.index(joke_id)
        similarity_scores = self.similarity_matrix[idx]
        top_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]
        recommended_jokes = [self.joke_ids[i] for i in top_indices]

        if not recommended_jokes:
            print(f"⚠️ No recommendations found for Joke ID {joke_id}. Returning an empty list.")
            return []  # Return an empty list if something went wrong

        return recommended_jokes

    def save_model(self, filepath):
        """
        💾 Saves the model as .npy files for the similarity matrix and joke IDs.

        Args:
            filepath (str): Directory path to save the model files.
        """
        os.makedirs(filepath, exist_ok=True)

        np.save(os.path.join(filepath, 'similarity_matrix.npy'), self.similarity_matrix)
        with open(os.path.join(filepath, 'joke_ids.json'), 'w') as f:
            json.dump(self.joke_ids, f)

        print(f"✅ Model saved at {filepath}")

    def load_model(self, filepath):
        """
        📦 Loads the saved similarity matrix and joke IDs.

        Args:
            filepath (str): Directory path to load the model files from.
        """
        self.similarity_matrix = np.load(os.path.join(filepath, 'similarity_matrix.npy'))

        with open(os.path.join(filepath, 'joke_ids.json'), 'r') as f:
            self.joke_ids = json.load(f)

        print(f"📦 Model loaded from {filepath}")