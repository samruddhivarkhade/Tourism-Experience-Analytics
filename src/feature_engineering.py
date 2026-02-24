import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


class FeatureEngineering:

    def __init__(self, processed_path):
        self.processed_path = processed_path
        self.master_path = os.path.join(processed_path, "master_dataset.csv")

    # --------------------------------------------------
    # Load Master Dataset
    # --------------------------------------------------
    def load_data(self):
        print("📂 Loading master dataset...")
        self.df = pd.read_csv(self.master_path)
        print("✅ Dataset loaded.")

    # --------------------------------------------------
    # Basic Feature Selection
    # --------------------------------------------------
    def select_features(self):

        print("🧠 Selecting important features...")

        columns_to_keep = [
            "VisitYear",
            "VisitMonth",
            "VisitMode",
            "Rating",
            "AttractionType",
            "Country",
            "Region",
            "Continent"
        ]

        self.df = self.df[columns_to_keep]

    # --------------------------------------------------
    # Encode Categorical Features
    # --------------------------------------------------
    def encode_features(self):

        print("🔄 Encoding categorical features...")

        self.label_encoders = {}

        categorical_columns = [
            "VisitMode",
            "AttractionType",
            "Country",
            "Region",
            "Continent"
        ]

        # 🔥 Create a copy for ML
        self.ml_df = self.df.copy()

        for col in categorical_columns:
            le = LabelEncoder()
            self.ml_df[col] = le.fit_transform(self.ml_df[col].astype(str))
            self.label_encoders[col] = le

        print("✅ Encoding completed.")

    # --------------------------------------------------
    # Save ML Dataset
    # --------------------------------------------------
    def save_processed(self):

        os.makedirs(self.processed_path, exist_ok=True)

        # 1️⃣ Save original readable dataset
        master_ml_path = os.path.join(self.processed_path, "ml_dataset.csv")
        self.ml_df.to_csv(master_ml_path, index=False)

        # 2️⃣ Save encoders
        encoder_path = os.path.join(self.processed_path, "label_encoders.pkl")
        import joblib
        joblib.dump(self.label_encoders, encoder_path)

        print(f"💾 ML dataset saved at: {master_ml_path}")
        print(f"💾 Encoders saved at: {encoder_path}")
    # --------------------------------------------------
    # Run Entire Feature Engineering
    # --------------------------------------------------
    def run(self):
        self.load_data()
        self.select_features()
        self.encode_features()
        self.save_processed()

        print("🎉 Feature engineering completed successfully.")

if __name__ == "__main__":
    fe = FeatureEngineering("processed")
    fe.run()