import polars as pl
import numpy as np
import mlflow
import mlflow.pyfunc
from models.baseline_model import BaselineModel
from models.collaborative_memory_model import CollaborativeMemoryModel
from models.collaborative_model_based import CollaborativeModelBased
from models.content_based_model import ContentBasedModel
from models.session_based_model import SessionBasedModel


def load_data(test_path="../data/processed/test_data.parquet", jokes_path="../data/processed/jokes_with_clusters.parquet"):
    print("\n📦 Loading test data...")
    test_df = pl.read_parquet(test_path)
    jokes_df = pl.read_parquet(jokes_path)
    print(f"✅ Test data loaded. Shape: {test_df.shape}, Joke data shape: {jokes_df.shape}")
    return test_df, jokes_df


def user_intersection(y_rel, y_rec, k=10):
    return len(set(y_rec[:k]).intersection(set(y_rel)))


def user_hitrate(y_rel, y_rec, k=10):
    return int(user_intersection(y_rel, y_rec, k) > 0)


def user_precision(y_rel, y_rec, k=10):
    return user_intersection(y_rel, y_rec, k) / k


def user_recall(y_rel, y_rec, k=10):
    return user_intersection(y_rel, y_rec, k) / len(set(y_rel)) if len(set(y_rel)) > 0 else 0


def user_ap(y_rel, y_rec, k=10):
    return np.sum([
        user_precision(y_rel, y_rec, idx + 1)
        for idx, item in enumerate(y_rec[:k]) if item in y_rel
    ]) / k


def user_rr(y_rel, y_rec, k=10):
    for idx, item in enumerate(y_rec[:k]):
        if item in y_rel:
            return 1 / (idx + 1)
    return 0


def user_ndcg(y_rel, y_rec, k=10):
    dcg = sum([1. / np.log2(idx + 2) for idx, item in enumerate(y_rec[:k]) if item in y_rel])
    idcg = sum([1. / np.log2(idx + 2) for idx, _ in enumerate(zip(y_rel, np.arange(k)))])
    return dcg / idcg if idcg > 0 else 0


def evaluate_all_models(test_df, jokes_df, k=10):
    """
    Load and evaluate all models, calculate all the metrics, and log them to MLflow.
    """
    with mlflow.start_run(run_name="Evaluation on Test Set"):
        # 1️⃣ Baseline Model
        print("\n🔹 Evaluating Baseline Model...")
        baseline_model = BaselineModel(ratings=test_df, items=jokes_df)
        baseline_model.load_model("../models/baseline_model.parquet")
        #evaluate_model(baseline_model, test_df, "Baseline", k)

        # 2️⃣ Collaborative Memory Model
        print("\n🔹 Evaluating Collaborative Memory-Based Model...")
        memory_model = CollaborativeMemoryModel()
        memory_model.load_model("../models/collaborative_memory_model.pkl")
        #evaluate_model(memory_model, test_df, "CollaborativeMemory", k)

        # 3️⃣ Collaborative Model-Based Model
        print("\n🔹 Evaluating Collaborative Model-Based Model...")
        collaborative_model_based = CollaborativeModelBased()
        collaborative_model_based.load_model("../models/collaborative_model_based.npz")
        #evaluate_model(collaborative_model_based, test_df, "CollaborativeModelBased", k)

        # 4️⃣ Content-Based Model
        print("\n🔹 Evaluating Content-Based Model...")
        content_model = ContentBasedModel()
        content_model.load_model("../models/content_based/")
        evaluate_model(content_model, test_df, "ContentBased", k)

        # 5️⃣ Session-Based Model
        print("\n🔹 Evaluating Session-Based Model...")
        session_model = SessionBasedModel()
        session_model.load_model("../models/session_gru.pth", "../models/session_cooccurrence.npy")
        #evaluate_model(session_model, test_df, "SessionBased", k)


def evaluate_model(model, test_df, model_name, k=10):
    """
    Evaluate a single model and log metrics to MLflow.
    """
    print(f"\n📈 Evaluating {model_name}...")
    hit_rates, precisions, recalls, ap_scores, rr_scores, ndcgs = [], [], [], [], [], []

    for user_id in test_df["userId"].unique():
        actual_jokes = test_df.filter(pl.col("userId") == user_id)["jokeId"].to_list()

        if len(actual_jokes) < 1:
            continue

        if hasattr(model, 'recommend'):
            recommended_jokes = model.recommend(user_id, top_n=k)
        else:
            recommended_jokes = model.predict()

        hit_rates.append(user_hitrate(actual_jokes, recommended_jokes, k))
        precisions.append(user_precision(actual_jokes, recommended_jokes, k))
        recalls.append(user_recall(actual_jokes, recommended_jokes, k))
        ap_scores.append(user_ap(actual_jokes, recommended_jokes, k))
        rr_scores.append(user_rr(actual_jokes, recommended_jokes, k))
        ndcgs.append(user_ndcg(actual_jokes, recommended_jokes, k))

    # Log aggregated metrics to MLflow
    mlflow.log_metric(f"{model_name}_HitRate", np.mean(hit_rates))
    mlflow.log_metric(f"{model_name}_Precision", np.mean(precisions))
    mlflow.log_metric(f"{model_name}_Recall", np.mean(recalls))
    mlflow.log_metric(f"{model_name}_AP", np.mean(ap_scores))
    mlflow.log_metric(f"{model_name}_RR", np.mean(rr_scores))
    mlflow.log_metric(f"{model_name}_NDCG", np.mean(ndcgs))

    print(f"✅ {model_name} evaluation complete!")


if __name__ == "__main__":
    test_df, jokes_df = load_data()
    evaluate_all_models(test_df, jokes_df)