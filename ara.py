import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Asteroid Hazard Prediction",
    page_icon="☄️",
    layout="wide"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the saved objects
@st.cache_resource
def load_model():
    with open('asteroid_classification_objects.pkl', 'rb') as f:
        return pickle.load(f)

# Function to make predictions
def predict_hazard(input_data, objects):
    # Scale the input data
    scaled_data = objects['scaler'].transform(input_data)
    
    # Get prediction, probability, and decision score
    prediction = objects['model'].predict(scaled_data)
    probability = objects['model'].predict_proba(scaled_data)
    decision_score = objects['model'].decision_function(scaled_data)  # Get decision score

    # Convert prediction to original label
    prediction_label = objects['label_encoder_hazardous'].inverse_transform(prediction)
    
    return prediction_label[0], probability[0], decision_score[0]  # Return decision score

def main():
    # Title and introduction
    st.title("🌠 Asteroid Risk Assessment")
    st.markdown("""
    This application uses machine learning to predict whether an asteroid poses a potential hazard to Earth
    based on its orbital characteristics.
    """)
    
    try:
        # Load model and objects
        objects = load_model()
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        # Input fields - using the feature columns from our model
        with col1:
            st.subheader("Orbital Parameters")
            inputs = {}
            
            for feature in objects['feature_columns']:
                inputs[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    help=f"Enter the value for {feature}"
                )
        
        # Create DataFrame from inputs
        input_df = pd.DataFrame([inputs])
        
        # Prediction button
        with col2:
            st.subheader("Prediction")
            if st.button("Predict Hazard Status"):
                # Make prediction
                prediction, probability, decision_score = predict_hazard(input_df, objects)

                # Calculate confidence
                confidence = max(probability) * 100
                
                # Create gauge chart for confidence
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig)

                # Display Decision Score Below Gauge Chart
                st.markdown(f"### Decision Score: `{decision_score:.4f}`")

                # Display prediction with appropriate styling
                if prediction:
                    st.error(f"⚠️ Prediction: POTENTIALLY HAZARDOUS")
                else:
                    st.success(f"✅ Prediction: NOT HAZARDOUS")
                
                # Display probability scores
                st.write("Probability Distribution:")
                prob_df = pd.DataFrame({
                    'Status': ['Not Hazardous', 'Hazardous'],
                    'Probability': probability
                })
                st.bar_chart(prob_df.set_index('Status'))
                
                # Additional information based on confidence
                if confidence < 20:
                    st.warning("⚠️ Low confidence prediction - consider gathering more data")
                elif confidence > 90:
                    st.info("🎯 High confidence prediction")
        
        # Add explanation section
        with st.expander("How to use this predictor"):
            st.markdown("""
            1. Enter the asteroid's orbital parameters in the input fields
            2. Click the 'Predict Hazard Status' button
            3. The system will display:
                - The predicted hazard status
                - A confidence score
                - Decision score
                - Probability distribution for both classes
            """)
        
        # Add feature importance information
        with st.expander("Feature Importance Information"):
            st.markdown("""
            The most important features for predicting asteroid hazard status are:
            1. Minimum Orbit Intersection
            2. Asteroid Magnitude
            3. Eccentricity
            
            Ensure these parameters are as accurate as possible for the best predictions.
            """)
    
    except Exception as e:
        st.error(f"""
        ⚠️ An error occurred while loading the model or making predictions.
        Error: {str(e)}
        
        Please ensure all model files are present and inputs are valid.
        """)

if __name__ == "__main__":
    main()
