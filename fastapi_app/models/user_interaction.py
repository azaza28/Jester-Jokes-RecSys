import mlflow

def log_user_interaction(user_id, group, recommendations, model_type):
    """Logs user interaction to MLFlow."""
    with mlflow.start_run(run_name=f"User-{user_id}"):
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("group", group)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("num_recommendations", len(recommendations))
        mlflow.log_artifact("path/to/user_interaction.log")