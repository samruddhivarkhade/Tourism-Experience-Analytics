import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionModel:

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

        if "Rating" not in self.df.columns:
            raise ValueError("Target column 'Rating' not found in dataset.")

        # Target
        y = self.df["Rating"]

        # Features (drop target only)
        X = self.df.drop(columns=["Rating"])

        # Save feature order for deployment
        self.feature_columns = X.columns.tolist()

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("✅ Data split completed.")
        print(f"   ➜ Training size: {self.X_train.shape}")
        print(f"   ➜ Testing size:  {self.X_test.shape}")

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    def train_model(self):
        print("🤖 Training Linear Regression model...")

        self.model = LinearRegression()

        self.model.fit(self.X_train, self.y_train)

        print("✅ Model training completed.")

    # --------------------------------------------------
    # Evaluate Model
    # --------------------------------------------------
    def evaluate_model(self):
        print("📊 Evaluating model...")

        y_pred = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(self.y_test, y_pred)

        print("\n📈 Model Performance:")
        print(f"   MAE  : {mae:.4f}")
        print(f"   MSE  : {mse:.4f}")
        print(f"   RMSE : {rmse:.4f}")
        print(f"   R²   : {r2:.4f}")

    # --------------------------------------------------
    # Save Model & Feature Order
    # --------------------------------------------------
    def save_model(self):
        print("💾 Saving model and feature metadata...")

        os.makedirs(self.model_path, exist_ok=True)

        model_file = os.path.join(self.model_path, "regression_model.pkl")
        features_file = os.path.join(self.model_path, "regression_features.pkl")

        # Save model
        joblib.dump(self.model, model_file, compress=3)

        # Save feature order
        joblib.dump(self.feature_columns, features_file)

        print(f"✅ Model saved at: {model_file}")
        print(f"✅ Feature order saved at: {features_file}")

    # --------------------------------------------------
    # Run Complete Pipeline
    # --------------------------------------------------
    def run(self):
        print("\n🚀 Starting Regression Pipeline...\n")

        self.load_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()

        print("\n🎉 Regression model completed successfully.\n")