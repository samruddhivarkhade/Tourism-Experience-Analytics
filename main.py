from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineering
from src.train_classification import ClassificationModel
from src.train_regression import RegressionModel
from src.recommender import TourismRecommender


# Run Data Pipeline
pipeline = DataPipeline(
    raw_path="data/raw",
    processed_path="data/processed"
)
pipeline.run()

# Run Feature Engineering
feature_engineer = FeatureEngineering(
    processed_path="data/processed"
)
feature_engineer.run()

# Train Classification Model
classifier = ClassificationModel(
    processed_path="data/processed",
    model_path="models"
)

classifier.run()

# Train Regression Model
regressor = RegressionModel(
    processed_path="data/processed",
    model_path="models"
)

regressor.run()

from src.recommender import TourismRecommender

recommender = TourismRecommender(processed_path="data/processed", model_path="models")
recommender.run()