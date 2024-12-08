import mlflow
import os

# Set tracking URI and experiment
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Model Registry")  # All models will be registered here

# Constants for model paths
BASELINE_MODEL_PATH = "../models/baseline_model/"
CONTENT_BASED_MODEL_PATH = "../models/content_based_model.pkl"
COLLABORATIVE_MEMORY_MODEL_PATH = "../models/collaborative_memory_model.pkl"
COLLABORATIVE_MODEL_BASED_PATH = "../models/collaborative_model_based/"
SESSION_BASED_MODEL_PATH = "../models/session_based_model/"


def register_baseline_model():
    try:
        with mlflow.start_run(run_name="Baseline Model", nested=True):
            mlflow.log_param("model_type", "Baseline")

            model_path = os.path.join(BASELINE_MODEL_PATH, 'baseline_model.pkl')
            mlflow.pyfunc.log_model(
                artifact_path="baseline_model",
                python_model=None,  # We don't need a custom model class here
                artifacts={"model_file": model_path}
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/baseline_model"
            mlflow.register_model(model_uri, "BaselineModel")
            print(f"✅ Baseline Model registered in MLflow.")
    except Exception as e:
        print(f"❌ Error registering Baseline Model: {e}")


def register_content_based_model():
    try:
        with mlflow.start_run(run_name="Content-Based Model", nested=True):
            mlflow.log_param("model_type", "Content-Based")

            model_path = CONTENT_BASED_MODEL_PATH
            mlflow.pyfunc.log_model(
                artifact_path="content_based_model",
                python_model=None,  # We don't need a custom model class here
                artifacts={"model_file": model_path}
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/content_based_model"
            mlflow.register_model(model_uri, "ContentBasedModel")
            print(f"✅ Content-Based Model registered in MLflow.")
    except Exception as e:
        print(f"❌ Error registering Content-Based Model: {e}")


def register_collaborative_memory_model():
    try:
        with mlflow.start_run(run_name="Collaborative Memory Model", nested=True):
            mlflow.log_param("model_type", "Collaborative Memory")

            model_path = COLLABORATIVE_MEMORY_MODEL_PATH
            mlflow.pyfunc.log_model(
                artifact_path="collaborative_memory_model",
                python_model=None,
                artifacts={"model_file": model_path}
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/collaborative_memory_model"
            mlflow.register_model(model_uri, "CollaborativeMemoryModel")
            print(f"✅ Collaborative Memory Model registered in MLflow.")
    except Exception as e:
        print(f"❌ Error registering Collaborative Memory Model: {e}")


def register_collaborative_model_based():
    try:
        with mlflow.start_run(run_name="Collaborative ALS Model", nested=True):
            mlflow.log_param("model_type", "Collaborative ALS")

            model_path = os.path.join(COLLABORATIVE_MODEL_BASED_PATH, 'model_factors.npz')
            mlflow.pyfunc.log_model(
                artifact_path="collaborative_als_model",
                python_model=None,
                artifacts={"model_file": model_path}
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/collaborative_als_model"
            mlflow.register_model(model_uri, "CollaborativeALSModel")
            print(f"✅ Collaborative Model-Based Model registered in MLflow.")
    except Exception as e:
        print(f"❌ Error registering Collaborative ALS Model: {e}")


def register_session_based_model():
    try:
        with mlflow.start_run(run_name="Session-Based Model", nested=True):
            mlflow.log_param("model_type", "Session-Based GRU")

            model_path = os.path.join(SESSION_BASED_MODEL_PATH, "session_gru.pth")
            mlflow.pyfunc.log_model(
                artifact_path="session_based_model",
                python_model=None,
                artifacts={"model_file": model_path}
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/session_based_model"
            mlflow.register_model(model_uri, "SessionBasedModel")
            print(f"✅ Session-Based Model registered in MLflow.")
    except Exception as e:
        print(f"❌ Error registering Session-Based Model: {e}")


if __name__ == "__main__":
    try:
        with mlflow.start_run(run_name="Model Registration", nested=False):
            register_baseline_model()
            register_content_based_model()
            register_collaborative_memory_model()
            register_collaborative_model_based()
            register_session_based_model()
            print("\n✅ All models successfully registered in MLflow!")
    except Exception as e:
        print(f"❌ Critical Error during Model Registration: {e}")
    finally:
        mlflow.end_run()