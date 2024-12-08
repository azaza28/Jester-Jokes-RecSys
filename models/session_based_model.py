import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import os
import json


class GRU4Rec(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim):
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
        self.model = None
        self.joke_index_map = None
        self.index_to_joke_id = None
        self.user_sessions = None

    def train(self, train_df, embedding_dim=64, hidden_dim=128, epochs=5, lr=0.001):
        print("\n🔹 Extracting user sessions for GRU training...")

        train_df = train_df.with_columns([
            pl.col("jokeId").cast(pl.Utf8).cast(pl.Categorical).alias("joke_idx")
        ])
        train_df = train_df.with_columns(pl.col("joke_idx").to_physical().alias("joke_idx_int"))

        joke_id_to_index = train_df['jokeId'].to_list()
        joke_index_to_id = train_df['joke_idx_int'].to_list()
        self.joke_index_map = dict(zip(joke_id_to_index, joke_index_to_id))
        self.index_to_joke_id = {v: k for k, v in self.joke_index_map.items()}

        self.user_sessions = self.extract_sessions(train_df)
        num_items = len(self.joke_index_map)

        print(f"🚀 Training GRU4Rec with {num_items} items (total unique jokes) ...")
        self.train_gru_model(self.user_sessions, num_items, embedding_dim, hidden_dim, epochs, lr)

    def extract_sessions(self, train_df, session_length=5):
        print("🔹 Extracting logical sessions from training data...")

        sessions = (
            train_df
            .sort(["userId", "joke_idx_int"])
            .groupby("userId")
            .agg(pl.col("joke_idx_int").alias("session"))
        )

        session_dict = {}
        for user_id, joke_indices in zip(sessions['userId'], sessions['session']):
            if len(joke_indices) > session_length:
                session_dict[user_id] = [int(x) for x in joke_indices[-session_length:]]
            else:
                session_dict[user_id] = [int(x) for x in joke_indices]

        print(f"✅ Extracted {len(session_dict)} sessions from training data.")
        return session_dict

    def train_gru_model(self, sessions, num_items, embedding_dim, hidden_dim, epochs, lr):
        model = GRU4Rec(num_items, embedding_dim, hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for user_id, session in sessions.items():
                if len(session) < 2:
                    continue

                session_tensor = torch.tensor([int(joke) for joke in session[:-1]], dtype=torch.long).unsqueeze(0)
                target = torch.tensor([int(joke) for joke in session[1:]], dtype=torch.long)

                optimizer.zero_grad()
                logits = model(session_tensor)
                loss = criterion(logits, target[-1].unsqueeze(0))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"🧮 Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        self.model = model
        print("✅ GRU4Rec model training complete.")

    def save_model(self, folder_path="../models/session_based_model/"):
        os.makedirs(folder_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(folder_path, 'session_gru.pth'))
        with open(os.path.join(folder_path, 'joke_index_map.json'), 'w') as f:
            json.dump(self.joke_index_map, f)
        with open(os.path.join(folder_path, 'user_sessions.json'), 'w') as f:
            json.dump(self.user_sessions, f)
        metadata = {
            "num_items": len(self.joke_index_map),
            "embedding_dim": self.model.embedding.embedding_dim,
            "hidden_dim": self.model.gru.hidden_size
        }
        with open(os.path.join(folder_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    @staticmethod
    def load_model(folder_path="../models/session_based_model/"):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"❌ Folder not found: {folder_path}")

        with open(os.path.join(folder_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        instance = SessionBasedModel()
        num_items = metadata['num_items']
        embedding_dim = metadata['embedding_dim']
        hidden_dim = metadata['hidden_dim']

        instance.model = GRU4Rec(num_items, embedding_dim, hidden_dim)
        instance.model.load_state_dict(torch.load(os.path.join(folder_path, 'session_gru.pth')))
        instance.model.eval()

        with open(os.path.join(folder_path, 'joke_index_map.json'), 'r') as f:
            instance.joke_index_map = json.load(f)
        with open(os.path.join(folder_path, 'user_sessions.json'), 'r') as f:
            instance.user_sessions = json.load(f)

        instance.index_to_joke_id = {v: k for k, v in instance.joke_index_map.items()}
        print(f"✅ Model loaded with {num_items} items, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
        return instance

    def recommend(self, user_id, train_df=None, top_n=10):
        if self.user_sessions is None or user_id not in self.user_sessions:
            if train_df is not None:
                user_session = train_df.filter(pl.col('userId') == user_id)['jokeId'].to_list()[-5:]
                if user_session:
                    joke_indices = [self.joke_index_map.get(str(joke_id)) for joke_id in user_session]
                    self.user_sessions[user_id] = [int(x) for x in joke_indices if x is not None]

        session = self.user_sessions.get(user_id, [])
        if len(session) > 5:
            session = session[-5:]

        session_tensor = torch.tensor([int(joke) for joke in session], dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(session_tensor)
            top_indices = torch.topk(logits, top_n).indices.squeeze().tolist()

        recommended_jokes = [self.index_to_joke_id[idx] for idx in top_indices if idx in self.index_to_joke_id]

        return recommended_jokes