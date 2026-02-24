import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


class TourismRecommender:

    def __init__(self, processed_path, model_path):
        self.processed_path = processed_path
        self.model_path = model_path
        self.data_path = os.path.join(processed_path, "ml_dataset.csv")

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------
    def load_data(self):
        print("📂 Loading dataset for recommendation...")
        self.df = pd.read_csv(self.data_path)
        print("✅ Dataset loaded.")

    # --------------------------------------------------
    # Prepare Feature Matrix
    # --------------------------------------------------
    def prepare_features(self):
        print("⚙ Encoding features...")

        self.feature_cols = [
            "VisitYear",
            "VisitMonth",
            "VisitMode",
            "AttractionType",
            "Country",
            "Region",
            "Continent"
        ]

        self.encoder = OneHotEncoder(handle_unknown="ignore")

        encoded = self.encoder.fit_transform(self.df[self.feature_cols])

        self.feature_matrix = encoded.toarray()

        print("✅ Feature matrix created.")

    # --------------------------------------------------
    # Compute Similarity
    # --------------------------------------------------
    

    # --------------------------------------------------
    # Save Components
    # --------------------------------------------------
    def save_recommender(self):
        os.makedirs(self.model_path, exist_ok=True)

        joblib.dump(self.encoder, os.path.join(self.model_path, "recommender_encoder.pkl"))
        joblib.dump(self.df, os.path.join(self.model_path, "recommender_data.pkl"))

        print("💾 Recommender saved successfully.")

    # --------------------------------------------------
    # Run Full Pipeline
    # --------------------------------------------------
    def run(self):
        print("\n🚀 Building Recommendation System...\n")

        self.load_data()
        self.prepare_features()
        self.save_recommender()

        print("\n🎉 Recommendation system ready!\n")

if __name__ == "__main__":
    recommender = TourismRecommender("processed", "models")
    recommender.run()