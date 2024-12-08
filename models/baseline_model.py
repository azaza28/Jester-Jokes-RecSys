import polars as pl
import joblib
import os


class BaselineModel:
    def __init__(self, ratings: pl.DataFrame = None, items: pl.DataFrame = None):
        self.ratings = ratings
        self.items = items
        self.all_jokes = None  # Store all jokes after training

    def train(self):
        """
        Train the baseline model to compute all jokes ranked by average rating.
        """
        print("🚀 Training Baseline Model...")

        # Extract user interaction counts
        user_interaction_counts = self.ratings.groupby("userId").count()

        # Filter out users with only one interaction
        valid_users = user_interaction_counts.filter(pl.col("count") > 1)["userId"]
        ratings_filtered = self.ratings.filter(pl.col("userId").is_in(valid_users))
        print("📊 Filtered valid users for training...")

        # Get all jokes ranked by average rating
        self.all_jokes = (
            ratings_filtered
            .groupby("jokeId")
            .agg(pl.col("rating").mean().alias("avg_rating"))
            .sort("avg_rating", descending=True)
            .select("jokeId")
            .to_series()
            .to_list()
        )
        print(f"✅ All jokes ranked successfully. Total jokes: {len(self.all_jokes)}")

    def save_model(self, folder_path: str):
        """
        Save the model to separate files.

        Args:
            folder_path (str): Directory path to save the model files.
        """
        os.makedirs(folder_path, exist_ok=True)

        print(f"💾 Saving Baseline Model to {folder_path}...")

        # Save Polars DataFrame (as Parquet)
        if self.ratings is not None:
            self.ratings.write_parquet(os.path.join(folder_path, 'ratings.parquet'))

        if self.items is not None:
            self.items.write_parquet(os.path.join(folder_path, 'items.parquet'))

        # Save other attributes with joblib
        joblib.dump({
            'all_jokes': self.all_jokes
        }, os.path.join(folder_path, 'baseline_model.pkl'))

        print(f"✅ Model saved successfully in {folder_path}")

    @staticmethod
    def load_model(folder_path: str):
        """
        Load the model from separate files in the specified directory.

        Args:
            folder_path (str): Directory path to load the model files.
        """
        print(f"📂 Loading Baseline Model from {folder_path}...")

        # Load Parquet files
        ratings = pl.read_parquet(os.path.join(folder_path, 'ratings.parquet'))
        items = pl.read_parquet(os.path.join(folder_path, 'items.parquet'))

        # Load other attributes from joblib file
        data = joblib.load(os.path.join(folder_path, 'baseline_model.pkl'))

        model = BaselineModel(ratings=ratings, items=items)
        model.all_jokes = data['all_jokes']

        print(f"✅ Model loaded successfully from {folder_path}.")
        return model

    def recommend(self, user_id=None, top_n=100):
        """
        Return the precomputed top-N jokes.
        """
        return self.all_jokes[:top_n]