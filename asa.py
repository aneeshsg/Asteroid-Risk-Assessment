import streamlit as st
import pandas as pd
import numpy as np
import pickle

def load_model_and_scaler():
    try:
        # Load the trained model
        with open('asteroid_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        # Load the scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure asteroid_svm_model.pkl and scaler.pkl are in the same directory as this script.")
        return None, None

def main():
    st.title("Asteroid Hazard Prediction")
    st.write("Enter asteroid parameters to predict if it's potentially hazardous")
    
    # Load model and scaler at startup
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    # Create input fields for features
    col1, col2 = st.columns(2)
    
    with col1:
        epoch = st.number_input("Epoch (TDB)", value=57800.0, format="%.1f")
        orbit_axis = st.number_input("Orbit Axis (AU)", value=2.0, format="%.4f")
        orbit_ecc = st.number_input("Orbit Eccentricity", value=0.5, min_value=0.0, max_value=1.0, format="%.4f")
        orbit_inc = st.number_input("Orbit Inclination (deg)", value=10.0, format="%.4f")
        peri_arg = st.number_input("Perihelion Argument (deg)", value=180.0, format="%.4f")
        
    with col2:
        mean_anom = st.number_input("Mean Anomoly (deg)", value=180.0, format="%.4f")
        peri_dist = st.number_input("Perihelion Distance (AU)", value=1.0, format="%.4f")
        aph_dist = st.number_input("Aphelion Distance (AU)", value=2.0, format="%.4f")
        orbital_period = st.number_input("Orbital Period (yr)", value=2.0, format="%.2f")
        
    col3, col4 = st.columns(2)
    
    with col3:
        moid = st.number_input("Minimum Orbit Intersection Distance (AU)", value=0.1, format="%.4f")
        orbit_ref = st.number_input("Orbital Reference", value=100, format="%d")
        ast_mag = st.number_input("Asteroid Magnitude", value=20.0, format="%.2f")
        
    # Create a prediction button
    if st.button("Predict Hazard"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Epoch (TDB)': [epoch],
            'Orbit Axis (AU)': [orbit_axis],
            'Orbit Eccentricity': [orbit_ecc],
            'Orbit Inclination (deg)': [orbit_inc],
            'Perihelion Argument (deg)': [peri_arg],
            'Mean Anomoly (deg)': [mean_anom],
            'Perihelion Distance (AU)': [peri_dist],
            'Aphelion Distance (AU)': [aph_dist],
            'Orbital Period (yr)': [orbital_period],
            'Minimum Orbit Intersection Distance (AU)': [moid],
            'Orbital Reference': [orbit_ref],
            'Asteroid Magnitude': [ast_mag],
        })
        
        try:
            # Scale the input data using the same scaler used during training
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            
            # Display result
            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error("⚠️ Potentially Hazardous Asteroid")
                st.write("This asteroid's characteristics suggest it could be potentially hazardous.")
            else:
                st.success("✅ Non-Hazardous Asteroid")
                st.write("This asteroid's characteristics suggest it is not potentially hazardous.")
                
            # Add confidence disclaimer
            st.info("Note: This is a simplified model and should not be used as the sole basis for actual asteroid risk assessment.")
            
        except Exception as e:
            st.error("An error occurred during prediction. Please check your input values.")
            st.write(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()