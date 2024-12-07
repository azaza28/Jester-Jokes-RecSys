import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")