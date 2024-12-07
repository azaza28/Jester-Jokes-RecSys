import mlflow


def log_baseline_model():
    with mlflow.start_run(run_name="Baseline Model"):
        mlflow.log_param("model_type", "Baseline")
        mlflow.log_artifact("../models/baseline_model.parquet", artifact_path="baseline_model")
        print(f"✅ Baseline Model logged to MLflow.")


def log_content_based_model():
    with mlflow.start_run(run_name="Content-Based Model"):
        mlflow.log_param("model_type", "Content-Based")
        mlflow.log_artifact("../models/content_based/similarity_matrix.npy", artifact_path="content_based")
        mlflow.log_artifact("../models/content_based/joke_ids.json", artifact_path="content_based")
        print(f"✅ Content-Based Model logged to MLflow.")


def log_collaborative_memory_model():
    with mlflow.start_run(run_name="Collaborative Memory Model"):
        mlflow.log_param("model_type", "Collaborative Memory")
        mlflow.log_artifact("../models/collaborative_memory_model.pkl", artifact_path="collaborative_memory_model")
        print(f"✅ Collaborative Memory-Based Model logged to MLflow.")


def log_collaborative_model_based():
    with mlflow.start_run(run_name="Collaborative ALS Model"):
        mlflow.log_param("model_type", "Collaborative ALS")
        mlflow.log_artifact("../models/collaborative_model_based.npz", artifact_path="collaborative_als_model")
        print(f"✅ Collaborative Model-Based Model logged to MLflow.")


def log_session_based_model():
    with mlflow.start_run(run_name="Session-Based Model"):
        mlflow.log_param("model_type", "Session-Based GRU")
        mlflow.log_artifact("../models/session_gru.pth", artifact_path="session_based_model")
        print(f"✅ Session-Based Model logged to MLflow.")


if __name__ == "__main__":
    log_baseline_model()
    log_content_based_model()
    log_collaborative_memory_model()
    log_collaborative_model_based()
    log_session_based_model()
