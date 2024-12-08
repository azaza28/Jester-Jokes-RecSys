import polars as pl
import numpy as np
import mlflow
import os
import json
from models.baseline_model import BaselineModel
from models.collaborative_memory_model import CollaborativeMemoryModel
from models.collaborative_model_based import CollaborativeModelBased
from models.content_based_model import ContentBasedModel
from models.session_based_model import SessionBasedModel

# Configure MLflow tracking URI and set experiment name
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Jester-Jokes-RecSys-Evaluation")


def load_data(test_path="../data/processed/test_features.parquet",
              train_path="../data/processed/train_features.parquet"):
    print("\n📦 Loading train and test data...")
    test_df = pl.read_parquet(test_path)
    train_df = pl.read_parquet(train_path)
    print(f"✅ Train data loaded. Shape: {train_df.shape}, Test data shape: {test_df.shape}")
    return train_df, test_df


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


def evaluate_all_models(train_df, test_df, k=10):
    """
    Load and evaluate all models, calculate all the metrics, and log them to MLflow.
    """
    with mlflow.start_run(run_name="Evaluation on Test Set"):
        models = {
            "Baseline": BaselineModel.load_model("../models/baseline_model/"),
            "CollaborativeMemory": CollaborativeMemoryModel.load_model("../models/collaborative_memory_model.pkl"),
            "CollaborativeModelBased": CollaborativeModelBased.load_model("../models/collaborative_model_based/"),
            "ContentBased": ContentBasedModel.load_model("../models/content_based_model.pkl"),
            "SessionBased": SessionBasedModel.load_model("../models/session_based_model")
        }

        for model_name, model in models.items():
            print(f"\n🔹 Evaluating {model_name}...")
            evaluate_model(model, test_df, train_df, model_name, k)


def evaluate_model(model, test_df, train_df, model_name, k=10):
    """
    Evaluate a single model, calculate average metrics, log them to MLflow, and print them.
    """
    print(f"\n📈 Evaluating {model_name}...")
    hit_rates, precisions, recalls, ap_scores, rr_scores, ndcgs = [], [], [], [], [], []

    with mlflow.start_run(run_name=f"Evaluation_{model_name}", nested=True):
        user_logs = {}

        try:
            for user_id in test_df["userId"].unique():
                try:
                    actual_jokes = test_df.filter(pl.col("userId") == user_id)["jokeId"].to_list()

                    if hasattr(model, 'recommend') and 'train_df' in model.recommend.__code__.co_varnames:
                        recommended_jokes = model.recommend(user_id, train_df=train_df, top_n=k)
                    else:
                        recommended_jokes = model.recommend(user_id, top_n=k)

                    if not recommended_jokes:
                        continue

                    metrics = {
                        "HitRate": user_hitrate(actual_jokes, recommended_jokes, k),
                        "Precision": user_precision(actual_jokes, recommended_jokes, k),
                        "Recall": user_recall(actual_jokes, recommended_jokes, k),
                        "AP": user_ap(actual_jokes, recommended_jokes, k),
                        "RR": user_rr(actual_jokes, recommended_jokes, k),
                        "NDCG": user_ndcg(actual_jokes, recommended_jokes, k)
                    }

                    user_logs[user_id] = {
                        "actual_jokes": actual_jokes,
                        "recommended_jokes": recommended_jokes,
                        "metrics": metrics
                    }

                    hit_rates.append(metrics["HitRate"])
                    precisions.append(metrics["Precision"])
                    recalls.append(metrics["Recall"])
                    ap_scores.append(metrics["AP"])
                    rr_scores.append(metrics["RR"])
                    ndcgs.append(metrics["NDCG"])

                except Exception as e:
                    print(f"❌ Error for user_id {user_id} in {model_name}: {e}")

            final_metrics = {
                f'HitRate{k}': np.mean(hit_rates) if hit_rates else 0,
                f'Precision{k}': np.mean(precisions) if precisions else 0,
                f'Recall{k}': np.mean(recalls) if recalls else 0,
                f'AP{k}': np.mean(ap_scores) if ap_scores else 0,
                f'RR{k}': np.mean(rr_scores) if rr_scores else 0,
                f'NDCG{k}': np.mean(ndcgs) if ndcgs else 0,
            }

            print(f"\n📊 Final Metrics for {model_name}:")
            for metric_name, metric_value in final_metrics.items():
                print(f"   📌 {metric_name}: {metric_value:.4f}")
                mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)

            log_path = f"../logs/{model_name}_user_logs.json"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(user_logs, f, indent=4)
            mlflow.log_artifact(log_path, artifact_path=f"{model_name}_logs")

        except Exception as e:
            print(f"❌ Critical Error in evaluating {model_name}: {e}")


if __name__ == "__main__":
    train_df, test_df = load_data()
    evaluate_all_models(train_df, test_df, k=30)