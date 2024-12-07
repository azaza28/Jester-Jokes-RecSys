# Use the official Python image as a base image
FROM python:3.10-slim

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5001
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db  # SQLite for local backend
ENV MLFLOW_ARTIFACT_STORE=./mlflow_artifacts  # Path to store artifacts locally

# Set the working directory inside the container
WORKDIR /app

# Copy all the project files into the container
COPY . .

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the MLflow server port
EXPOSE 5001

# Command to start FastAPI and MLflow at the same time
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri $MLFLOW_BACKEND_STORE_URI --default-artifact-root $MLFLOW_ARTIFACT_STORE & uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000 --reload"]