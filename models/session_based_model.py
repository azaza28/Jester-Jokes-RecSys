import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import os


class GRU4Rec(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim):
        """
        📘 GRU-based session-based recommendation model.

        Args:
            num_items (int): Number of unique items in the dataset.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the GRU hidden state.
        """
        super(GRU4Rec, self).__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_items)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        logits = self.output_layer(output[:, -1, :])  # Only use the last output
        return logits


class SessionBasedModel:
    def __init__(self):
        """
        📘 Session-based recommendation model (GRU + Co-occurrence)
        """
        self.model = None
        self.co_occurrence_matrix = None

    def train(self, train_df, embedding_dim=64, hidden_dim=128, epochs=5, lr=0.001):
        """
        🚀 Train the GRU4Rec model using session-based training.

        Args:
            train_df (pl.DataFrame): DataFrame with columns ['userId', 'jokeId'].
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the GRU hidden state.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        print("\n🔹 Extracting user sessions for GRU training...")

        # 🔥 Extract user sessions
        sessions = self.extract_sessions(train_df)
        num_items = train_df["jokeId"].max() + 1  # Number of unique items

        print(f"📦 Extracted {len(sessions)} sessions for training.")

        # 🚀 Train GRU4Rec
        print(f"🚀 Training GRU4Rec with {num_items} items...")
        self.train_gru_model(sessions, num_items, embedding_dim, hidden_dim, epochs, lr)

    def extract_sessions(self, train_df):
        """
        📦 Extract user sessions from the train DataFrame.

        Args:
            train_df (pl.DataFrame): The training DataFrame with columns ['userId', 'jokeId'].

        Returns:
            list: List of sessions, where each session is a list of joke IDs interacted by the user.
        """
        print("🔹 Extracting sessions from training data...")
        sessions = (
            train_df
            .sort(["userId", "jokeId"])  # Sort by userId and jokeId (or use timestamp if you have it)
            .groupby("userId")
            .agg(pl.col("jokeId").alias("session"))
        )

        session_list = sessions["session"].to_list()
        print(f"✅ Extracted {len(session_list)} sessions from training data.")
        return session_list

    def train_gru_model(self, sessions, num_items, embedding_dim, hidden_dim, epochs, lr):
        """
        🚀 Train the GRU4Rec model.

        Args:
            sessions (list): List of user sessions.
            num_items (int): Number of unique items.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the GRU hidden state.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        model = GRU4Rec(num_items, embedding_dim, hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for session in sessions:
                if len(session) < 2:  # 🚫 Skip short sessions
                    continue
                session_tensor = torch.tensor(session[:-1]).unsqueeze(0)  # Shape: (1, T-1)
                target = torch.tensor(session[1:])  # Target for the next item

                optimizer.zero_grad()
                logits = model(session_tensor)  # Predict for the entire sequence
                loss = criterion(logits, target[-1].unsqueeze(0))  # 🎯 Only predict the last item
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"🧮 Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        self.model = model
        print("✅ GRU4Rec model training complete.")

    def save_model(self, model_path="../models/gru4rec.pth", co_occurrence_path="../models/cooccurrence_matrix.npy"):
        """
        💾 Save the GRU model and co-occurrence matrix.

        Args:
            model_path (str): Path to save the GRU model.
            co_occurrence_path (str): Path to save the co-occurrence matrix.
        """
        if self.model is not None:
            torch.save(self.model.state_dict(), model_path)
            print(f"✅ GRU4Rec model saved at {model_path}")
        else:
            print("⚠️ No GRU model to save.")

        if self.co_occurrence_matrix is not None:
            np.save(co_occurrence_path, self.co_occurrence_matrix)
            print(f"✅ Co-occurrence matrix saved at {co_occurrence_path}")
        else:
            print("⚠️ No co-occurrence matrix to save.")

    def load_model(self, model_path="../models/gru4rec.pth", num_items=None, embedding_dim=64, hidden_dim=128):
        """
        📦 Load the GRU4Rec model.

        Args:
            model_path (str): Path to the GRU model file.
            num_items (int): Number of unique items (needed for model architecture).
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the GRU hidden state.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found at {model_path}")

        self.model = GRU4Rec(num_items, embedding_dim, hidden_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"✅ GRU4Rec model loaded successfully from {model_path}")

    def predict_next_item(self, session):
        """
        🔮 Predict the next item for a given session.

        Args:
            session (list): List of joke IDs representing the user's session.

        Returns:
            int: Predicted next joke ID.
        """
        if self.model is None:
            raise ValueError("❌ Model is not loaded. Call load_model() first.")

        session_tensor = torch.tensor(session).unsqueeze(0)  # Shape: (1, T)
        logits = self.model(session_tensor)
        predicted_item = torch.argmax(logits, dim=1).item()

        print(f"🔮 Predicted next item for session {session}: {predicted_item}")
        return predicted_item