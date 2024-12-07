import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.onnx
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from models.baseline_model import BaselineModel
from models.content_based_model import ContentBasedModel
from models.collaborative_memory_model import CollaborativeMemoryModel
from models.collaborative_model_based import CollaborativeModelBased
from models.session_based_model import SessionBasedModel
import polars as pl



def train_and_log_baseline(train_df, items_df):
    print("\n🔹 Training and Logging Baseline Model to MLflow...")
    with mlflow.start_run(run_name="Baseline Model"):
        baseline_model = BaselineModel(ratings=train_df, items=items_df)
        baseline_model.train()

        # Log modelkill -9 $(lsof -ti:3000) as an artifact in MLflow
        mlflow.log_param("model_type", "Baseline")
        mlflow.log_param("num_jokes", len(baseline_model.top_100_jokes))
        baseline_model_path = "models/baseline_model.parquet"
        baseline_model.save_model(baseline_model_path)

        mlflow.log_artifact(baseline_model_path)
        print("✅ Baseline Model saved and logged to MLflow")


def train_and_log_content_based(jokes_df):
    print("\n🔹 Training and Logging Content-Based Model to MLflow...")
    with mlflow.start_run(run_name="Content-Based Model"):
        content_model = ContentBasedModel()
        embeddings = jokes_df['embeddings'].to_list()
        joke_ids = jokes_df['jokeId'].to_list()

        content_model.train(embeddings, joke_ids)

        mlflow.log_param("model_type", "Content-Based")
        mlflow.log_param("num_jokes", len(joke_ids))

        model_path = "models/content_based/"
        content_model.save_model(model_path)

        mlflow.log_artifact(model_path)
        print("✅ Content-Based Model saved and logged to MLflow")


def train_and_log_collaborative_memory(train_df):
    print("\n🔹 Training and Logging Collaborative Memory Model to MLflow...")
    with mlflow.start_run(run_name="Collaborative Memory Model"):
        memory_model = CollaborativeMemoryModel()
        memory_model.train(train_df)

        mlflow.log_param("model_type", "Collaborative Memory-Based")

        model_path = "models/collaborative_memory.pkl"
        memory_model.save_model(model_path)

        mlflow.log_artifact(model_path)
        print("✅ Collaborative Memory Model saved and logged to MLflow")


def train_and_log_collaborative_model_based(train_df):
    print("\n🔹 Training and Logging Collaborative Model-Based to MLflow...")
    with mlflow.start_run(run_name="Collaborative Model-Based"):
        model_based = CollaborativeModelBased()
        model_based.train(train_df)

        mlflow.log_param("model_type", "Collaborative Model-Based")

        model_path = "models/collaborative_model_based.npz"
        model_based.save_model(model_path)

        mlflow.log_artifact(model_path)
        print("✅ Collaborative Model-Based saved and logged to MLflow")


def train_and_log_session_based(train_df):
    print("\n🔹 Training and Logging Session-Based Model to MLflow...")
    with mlflow.start_run(run_name="Session-Based Model"):
        session_model = SessionBasedModel()
        session_model.train(train_df)

        mlflow.log_param("model_type", "Session-Based")

        model_path = "models/session_gru.pth"
        session_model.save_model(model_path)

        mlflow.log_artifact(model_path)
        print("✅ Session-Based Model saved and logged to MLflow")


if __name__ == "__main__":
    train_df = pl.read_csv("../data/processed/train_data.csv")
    jokes_df = pl.read_parquet("../data/processed/jokes_with_clusters.parquet")

    train_and_log_baseline(train_df, jokes_df)
    train_and_log_content_based(jokes_df)
    train_and_log_collaborative_memory(train_df)
    train_and_log_collaborative_model_based(train_df)
    train_and_log_session_based(train_df)