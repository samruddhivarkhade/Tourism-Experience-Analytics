import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

st.title("🌍 Tourism Experience Analytics")
st.write("Predict Visit Mode, Rating & Get Recommendations")

# =====================================================
# Load Models & Data
# =====================================================
@st.cache_resource
def load_models():
    classification_model = joblib.load("models/classification_model.pkl")
    regression_model = joblib.load("models/regression_model.pkl")
    recommender_encoder = joblib.load("models/recommender_encoder.pkl")
    recommender_data = joblib.load("models/recommender_data.pkl")
    label_encoders = joblib.load("data/processed/label_encoders.pkl")

    return (
        classification_model,
        regression_model,
        recommender_encoder,
        recommender_data,
        label_encoders
    )

(
    classification_model,
    regression_model,
    recommender_encoder,
    recommender_data,
    label_encoders
) = load_models()

# Extract individual encoders
visit_mode_encoder = label_encoders["VisitMode"]
attraction_encoder = label_encoders["AttractionType"]
country_encoder = label_encoders["Country"]
region_encoder = label_encoders["Region"]
continent_encoder = label_encoders["Continent"]

# =====================================================
# Input Section
# =====================================================
st.subheader("📌 Enter Visit Details")

visit_year = st.number_input("Visit Year", min_value=2000, max_value=2030, value=2013)
visit_month = st.number_input("Visit Month", min_value=1, max_value=12, value=1)

# Use dropdowns instead of numbers (professional UI)
visit_mode_options = visit_mode_encoder.classes_
attraction_options = attraction_encoder.classes_
country_options = country_encoder.classes_
region_options = region_encoder.classes_
continent_options = continent_encoder.classes_

attraction_type = st.selectbox("Attraction Type", attraction_options)
country = st.selectbox("Country", country_options)
region = st.selectbox("Region", region_options)
continent = st.selectbox("Continent", continent_options)

# Encode inputs
encoded_attraction = attraction_encoder.transform([attraction_type])[0]
encoded_country = country_encoder.transform([country])[0]
encoded_region = region_encoder.transform([region])[0]
encoded_continent = continent_encoder.transform([continent])[0]

# =====================================================
# Prediction
# =====================================================
if st.button("Predict"):

    base_input = pd.DataFrame([{
        "VisitYear": visit_year,
        "VisitMonth": visit_month,
        "VisitMode": 0,  # placeholder (not used for classification input)
        "AttractionType": encoded_attraction,
        "Country": encoded_country,
        "Region": encoded_region,
        "Continent": encoded_continent
    }])

    # Classification (Predict Visit Mode)
    class_input = base_input.drop(columns=["VisitMode"])
    predicted_mode = classification_model.predict(class_input)[0]
    decoded_mode = visit_mode_encoder.inverse_transform([predicted_mode])[0]

    # Regression (Predict Rating)
    base_input["VisitMode"] = predicted_mode
    predicted_rating = regression_model.predict(base_input)[0]

    st.success("Prediction Complete!")

    st.subheader("🧭 Visit Mode")
    st.success(decoded_mode)

    st.subheader("⭐ Rating")
    st.success(f"{round(predicted_rating, 2)} / 5")

    # =====================================================
    # Recommendation System
    # =====================================================
    st.subheader("🔥 Recommended Experiences")

    rec_input = pd.DataFrame([{
        "VisitYear": visit_year,
        "VisitMonth": visit_month,
        "VisitMode": predicted_mode,
        "AttractionType": encoded_attraction,
        "Country": encoded_country,
        "Region": encoded_region,
        "Continent": encoded_continent
    }])

    # Encode for similarity
    rec_encoded = recommender_encoder.transform(rec_input)

    recommender_features_only = recommender_data.drop(columns=["Rating"], errors="ignore")

    full_matrix = recommender_encoder.transform(
        recommender_features_only
    )

    similarities = cosine_similarity(rec_encoded, full_matrix)
    similarity_scores = list(enumerate(similarities[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

    if not similarity_scores:
        st.warning("No recommendations found.")
    else:
        indices = [i[0] for i in similarity_scores]
        recommendations = recommender_data.iloc[indices]

        for _, row in recommendations.iterrows():

            decoded_attraction = attraction_encoder.inverse_transform(
                [row["AttractionType"]]
            )[0]

            decoded_country = country_encoder.inverse_transform(
                [row["Country"]]
            )[0]

            decoded_visit_mode = visit_mode_encoder.inverse_transform(
                [row["VisitMode"]]
            )[0]

            st.write(
                f"⭐ {round(row['Rating'], 2)} | "
                f"{decoded_attraction} - "
                f"{decoded_country} ({decoded_visit_mode})"
            )