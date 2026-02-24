import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class ClassificationModel:

    def __init__(self, processed_path, model_path):
        self.processed_path = processed_path
        self.model_path = model_path
        self.data_path = os.path.join(processed_path, "ml_dataset.csv")

        self.model = None
        self.feature_columns = None

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------
    def load_data(self):
        print("📂 Loading ML dataset...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded. Shape: {self.df.shape}")

    # --------------------------------------------------
    # Prepare Features & Target
    # --------------------------------------------------
    def prepare_data(self):
        print("⚙ Preparing features and target...")

        if "VisitMode" not in self.df.columns:
            raise ValueError("Target column 'VisitMode' not found.")

        # Target
        y = self.df["VisitMode"]

        # Drop VisitMode AND Rating
        X = self.df.drop(columns=["VisitMode", "Rating"])

        # Save feature order
        self.feature_columns = X.columns.tolist()

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("✅ Data split completed.")

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    def train_model(self):
        print("🤖 Training Logistic Regression model...")

        self.model = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial"
        )

        self.model.fit(self.X_train, self.y_train)

        print("✅ Model training completed.")

    # --------------------------------------------------
    # Evaluate Model
    # --------------------------------------------------
    def evaluate_model(self):
        print("📊 Evaluating model...")

        y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)

        print("\n📈 Model Performance:")
        print(f"   Accuracy: {acc:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(self.y_test, y_pred))

    # --------------------------------------------------
    # Save Model & Feature Order
    # --------------------------------------------------
    def save_model(self):
        print("💾 Saving model and feature metadata...")

        os.makedirs(self.model_path, exist_ok=True)

        model_file = os.path.join(self.model_path, "classification_model.pkl")
        features_file = os.path.join(self.model_path, "classification_features.pkl")

        joblib.dump(self.model, model_file, compress=3)
        joblib.dump(self.feature_columns, features_file)

        print(f"✅ Model saved at: {model_file}")
        print(f"✅ Feature order saved at: {features_file}")

    # --------------------------------------------------
    # Run Pipeline
    # --------------------------------------------------
    def run(self):
        print("\n🚀 Starting Classification Pipeline...\n")

        self.load_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()

        print("\n🎉 Classification model completed successfully.\n")