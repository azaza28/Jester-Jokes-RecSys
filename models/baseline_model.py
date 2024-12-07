import polars as pl


class BaselineModel:
    def __init__(self, ratings: pl.DataFrame, items: pl.DataFrame):
        self.ratings = ratings
        self.items = items
        self.top_100_jokes = None  # Store the top 100 jokes after training

    def train(self):
        """
        Train the baseline model to compute the top 100 jokes based on average rating.
        """
        print("🚀 Training Baseline Model...")

        # Extract user interaction counts
        user_interaction_counts = self.ratings.groupby("userId").count()

        # Filter out users with only one interaction
        valid_users = user_interaction_counts.filter(pl.col("count") > 1)["userId"]
        ratings_filtered = self.ratings.filter(pl.col("userId").is_in(valid_users))
        print("📊 Filtered valid users for training...")

        # Get top 100 jokes based on average rating
        self.top_100_jokes = (
            ratings_filtered
            .groupby("jokeId")
            .agg(pl.col("rating").mean().alias("avg_rating"))
            .sort("avg_rating", descending=True)
            .head(100)
            .select("jokeId")
            .to_series()
            .to_list()
        )
        print(f"✅ Top 100 Jokes calculated successfully. Total jokes: {len(self.top_100_jokes)}")

    def save_model(self, filepath: str):
        """
        Save the top 100 jokes as a Parquet or JSON file.
        """
        print(f"💾 Saving model to {filepath}...")
        df = pl.DataFrame({"top_100_jokes": self.top_100_jokes})
        if filepath.endswith(".parquet"):
            df.write_parquet(filepath)
        elif filepath.endswith(".json"):
            df.write_json(filepath)
        else:
            raise ValueError("❌ Unsupported file format. Use .parquet or .json")
        print(f"✅ Model saved successfully at {filepath}")

    def load_model(self, filepath: str):
        """
        Load the top 100 jokes from a Parquet or JSON file.
        """
        print(f"📂 Loading model from {filepath}...")
        if filepath.endswith(".parquet"):
            self.top_100_jokes = pl.read_parquet(filepath)["top_100_jokes"].to_list()
        elif filepath.endswith(".json"):
            self.top_100_jokes = pl.read_json(filepath)["top_100_jokes"].to_list()
        else:
            raise ValueError("❌ Unsupported file format. Use .parquet or .json")
        print(f"✅ Model loaded successfully. Total jokes: {len(self.top_100_jokes)}")

    def predict(self):
        """
        Return the precomputed top 100 jokes.
        """
        print("🎉 Returning top 100 precomputed jokes...")
        return self.top_100_jokes