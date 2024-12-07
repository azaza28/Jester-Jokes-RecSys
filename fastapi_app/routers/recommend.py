from fastapi import APIRouter, HTTPException
import polars as pl  # Assuming you're using Polars for DataFrame
from models.baseline_model import BaselineModel
from models.content_based_model import ContentBasedModel
from models.collaborative_memory_model import CollaborativeMemoryModel
from models.collaborative_model_based import CollaborativeModelBased
from models.session_based_model import SessionBasedModel
import numpy as np
from typing import List, Any

router = APIRouter(
    prefix="/api",
    tags=["API"]
)

# 🔥 Step 1: Load required data for models
print("📦 Loading required data...")

try:
    ratings_path = "../data/processed/train_data.parquet"
    items_path = "../data/processed/jokes_with_clusters.parquet"

    ratings = pl.read_parquet(ratings_path)
    items = pl.read_parquet(items_path)

    print(f"✅ Ratings loaded with shape {ratings.shape} from {ratings_path}")
    print(f"✅ Items loaded with shape {items.shape} from {items_path}")
except Exception as e:
    print(f"❌ Failed to load ratings or items data. Error: {str(e)}")

# 🔥 Step 2: Initialize all models
try:
    baseline_model = BaselineModel(ratings=ratings, items=items)
    content_based_model = ContentBasedModel()
    collaborative_memory_model = CollaborativeMemoryModel()
    collaborative_model_based = CollaborativeModelBased()
    session_based_model = SessionBasedModel()

    # Load pre-trained models if available
    content_based_model.load_model("../models/content_based/")
    collaborative_memory_model.load_model("../models/collaborative_memory_model.pkl")
    collaborative_model_based.load_model("../models/collaborative_filtering.npz")
    session_based_model.load_model("../models/session_gru.pth")

    print(f"✅ Models initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize models. Error: {str(e)}")



@router.get("/recommend/{user_id}")
def get_recommendations(user_id: int, joke_id: int = None):
    """Endpoint to get recommendations for a specific user."""
    group = "A"  # Assume a simple group assignment for now

    if group == 'A':
        recommendations = baseline_model.predict()
    elif group == 'B':
        recommendations = content_based_model.recommend(joke_id)
    elif group == 'C':
        recommendations = collaborative_memory_model.recommend(user_id)
    elif group == 'D':
        recommendations = collaborative_model_based.recommend(user_id)
    elif group == 'E':
        recommendations = session_based_model.recommend_from_session(user_id)
    else:
        raise HTTPException(status_code=500, detail="Invalid group assignment.")

    return {"user_id": user_id, "group": group, "recommendations": recommendations}


@router.post("/train/{model_name}")
def train_model(model_name: str):
    """Train a specific model by name (baseline, content-based, collaborative, etc.)"""
    try:
        if model_name == 'baseline':
            print(f"🚀 Training Baseline Model...")
            baseline_model.train()
            baseline_model.save_model("../models/baseline_model.parquet")
        elif model_name == 'content_based':
            print(f"🚀 Training Content-Based Model...")
            joke_ids = items["jokeId"].to_list()
            embeddings = pl.from_pandas(items.select("embeddings").to_pandas())["embeddings"].to_list()
            embeddings = np.vstack(embeddings)
            content_based_model.train(embeddings, joke_ids)
            content_based_model.save_model("../models/content_based/")
        elif model_name == 'collaborative_memory':
            print(f"🚀 Training Collaborative Memory Model...")
            collaborative_memory_model.train(ratings)
            collaborative_memory_model.save_model("../models/collaborative_memory_model.pkl")
        elif model_name == 'collaborative_model_based':
            print(f"🚀 Training Collaborative Model-Based Model...")
            collaborative_model_based.train(ratings)
            collaborative_model_based.save_model("../models/collaborative_filtering.npz")
        elif model_name == 'session_based':
            print(f"🚀 Training Session-Based Model...")
            session_based_model.train(ratings)
            session_based_model.save_model("../models/session_gru.pth", "../models/session_cooccurrence.npy")
        else:
            raise HTTPException(status_code=400, detail="Invalid model name.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed for {model_name}. Error: {str(e)}")

    return {"message": f"Training for {model_name} model started successfully."}


@router.get("/test/{model_name}")
def test_model(model_name: str):
    """Test a specific model by name (baseline, content-based, collaborative, etc.)"""
    try:
        predictions = []  # Replace with the actual predictions from the model
        relevant_items = [1, 2, 3]  # Replace with the actual relevant items
        test_results = evaluate_predictions(predictions, relevant_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Testing failed for {model_name}. Error: {str(e)}")

    return {"message": f"Testing for {model_name} model completed successfully.", "test_results": test_results}


def user_intersection(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> int:
    """
    Count the intersection of relevant items and recommended items.

    Args:
        y_rel (List[Any]): List of relevant items.
        y_rec (List[Any]): List of recommended items.
        k (int): Number of recommendations to consider.

    Returns:
        int: The number of items in the intersection of y_rel and y_rec.
    """
    return len(set(y_rec[:k]).intersection(set(y_rel)))


def user_hitrate(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> int:
    """
    Calculate HitRate@K.

    Args:
        y_rel (List[Any]): List of relevant items.
        y_rec (List[Any]): List of recommended items.
        k (int): Number of recommendations to consider.

    Returns:
        int: 1 if at least one relevant item is in the top-K recommended items, otherwise 0.
    """
    return int(user_intersection(y_rel, y_rec, k) > 0)


def user_precision(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Calculate Precision@K.

    Args:
        y_rel (List[Any]): List of relevant items.
        y_rec (List[Any]): List of recommended items.
        k (int): Number of recommendations to consider.

    Returns:
        float: Precision@K, the percentage of relevant items in the recommendations.
    """
    if k == 0:
        return 0.0
    return user_intersection(y_rel, y_rec, k) / k


def user_recall(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Calculate Recall@K.

    Args:
        y_rel (List[Any]): List of relevant items.
        y_rec (List[Any]): List of recommended items.
        k (int): Number of recommendations to consider.

    Returns:
        float: Recall@K, the fraction of relevant items that are in the recommendations.
    """
    if len(set(y_rel)) == 0:
        return 0.0
    return user_intersection(y_rel, y_rec, k) / len(set(y_rel))


def user_ap(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Calculate Average Precision (AP@K).

    Args:
        y_rel (List[Any]): List of relevant items.
        y_rec (List[Any]): List of recommended items.
        k (int): Number of recommendations to consider.

    Returns:
        float: Average precision for the user recommendations.
    """
    precisions = [
        user_precision(y_rel, y_rec, idx + 1)
        for idx, item in enumerate(y_rec[:k]) if item in y_rel
    ]
    if not precisions:
        return 0.0
    return sum(precisions) / k


def user_rr(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Calculate Reciprocal Rank (RR@K).

    Args:
        y_rel (List[Any]): List of relevant items.
        y_rec (List[Any]): List of recommended items.
        k (int): Number of recommendations to consider.

    Returns:
        float: Reciprocal rank of the first relevant item in the recommended list.
    """
    for idx, item in enumerate(y_rec[:k]):
        if item in y_rel:
            return 1 / (idx + 1)
    return 0.0


def user_ndcg(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        y_rel (List[Any]): List of relevant items.
        y_rec (List[Any]): List of recommended items.
        k (int): Number of recommendations to consider.

    Returns:
        float: NDCG@K, the normalized measure of ranking quality.
    """
    dcg = sum([1.0 / np.log2(idx + 2) for idx, item in enumerate(y_rec[:k]) if item in y_rel])
    idcg = sum([1.0 / np.log2(idx + 2) for idx, _ in enumerate(zip(y_rel, np.arange(k)))])
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_predictions(predictions, relevant_items=[1, 2, 3]):
    """
    Evaluate the performance of predictions using multiple metrics.

    Args:
        predictions (list): List of recommended items.
        relevant_items (list): List of ground-truth relevant items.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        "hitrate": user_hitrate(relevant_items, predictions, k=10),
        "precision": user_precision(relevant_items, predictions, k=10),
        "recall": user_recall(relevant_items, predictions, k=10),
        "ndcg": user_ndcg(relevant_items, predictions, k=10),
        "ap": user_ap(relevant_items, predictions, k=10),
        "rr": user_rr(relevant_items, predictions, k=10)
    }
    return metrics

@router.get("/help")
def get_help():
    """List of all API endpoints."""
    endpoints = [
        "/api/recommend/{user_id}",
        "/api/train/{model_name}",
        "/api/test/{model_name}",
        "/api/insights/{model_name}",
        "/api/help",
        "/api/health"
    ]
    return {"endpoints": endpoints}


@router.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "API is up and running!"}


@router.get("/insights/{model_name}")
def model_insights(model_name: str):
    """Get model-specific insights (like number of users, items, etc.)"""
    if model_name == 'baseline':
        insights = {"description": "Simple Top-100 model", "num_jokes": 100}
    elif model_name == 'content_based':
        insights = {"description": "Content-Based model using embeddings",
                    "num_jokes": len(content_based_model.joke_ids)}
    elif model_name == 'collaborative_memory':
        insights = {"description": "Collaborative Memory model (User-User or Item-Item similarity)",
                    "num_users": len(collaborative_memory_model.user_ids)}
    elif model_name == 'collaborative_model_based':
        insights = {"description": "Collaborative Model-Based (ALS) model",
                    "num_users": len(collaborative_model_based.user_ids)}
    elif model_name == 'session_based':
        insights = {"description": "Session-Based GRU model", "num_sessions": "Dynamic during training"}
    else:
        raise HTTPException(status_code=400, detail="Invalid model name.")
    return insights