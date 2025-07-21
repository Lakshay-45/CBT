"""
A Streamlit web application to deploy the trained RandomForest model for
predicting wine class based on user input.
"""

import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_wine

# --- Page Configuration ---
st.set_page_config(
    page_title="Wine Class Prediction",
    page_icon="🍷",
    layout="wide"
)

# --- Model and Data Loading ---
@st.cache_resource
def load_model_and_data():
    """Load the saved model and dataset for feature information."""
    model = joblib.load('wine_quality_model.joblib')
    wine_data = load_wine()
    feature_names = wine_data.feature_names
    target_names = wine_data.target_names
    # Create a dataframe for min/max values for sliders
    df = pd.DataFrame(wine_data.data, columns=feature_names)
    return model, df, feature_names, target_names

model, wine_df, feature_names, target_names = load_model_and_data()

# --- Application UI ---
st.title("🍷 Wine Class Prediction")
st.markdown("""
This application predicts the class of wine based on its chemical properties.
Use the sliders in the sidebar to input the feature values.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Input Wine Features")

def get_user_input():
    """Create sliders in the sidebar and return user inputs as a DataFrame."""
    inputs = {}
    for feature in feature_names:
        min_val = float(wine_df[feature].min())
        max_val = float(wine_df[feature].max())
        mean_val = float(wine_df[feature].mean())
        inputs[feature] = st.sidebar.slider(
            label=feature.replace('_', ' ').capitalize(),
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    return pd.DataFrame([inputs])

user_input_df = get_user_input()

# --- Prediction and Output ---
st.header("Prediction Results")

# Predict the class
prediction = model.predict(user_input_df)
prediction_proba = model.predict_proba(user_input_df)

# Display the prediction
predicted_class_name = target_names[prediction[0]].replace('_', ' ').capitalize()
st.success(f"**Predicted Wine Class:** {predicted_class_name}")

# --- Visualizations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input:")
    st.dataframe(user_input_df.T.rename(columns={0: 'Value'}))

with col2:
    st.subheader("Prediction Probability:")
    # Create a DataFrame for the probabilities
    proba_df = pd.DataFrame(
        prediction_proba,
        columns=[name.capitalize() for name in target_names],
        index=["Probability"]
    ).T
    st.bar_chart(proba_df)

st.info("This app uses a pre-trained RandomForestClassifier to make predictions.")