import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="GlucoRiskAI",
    page_icon="🩺",
    layout="centered",
)

st.title("GlucoRiskAI")
st.subheader("Diabetes Risk Profiling")
st.write(
    "Enter the patient's information to predict their "
    "project-defined risk category."
)

st.divider()

with st.form("patient_form"):
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            max_value=300.0,
            value=130.0,
            step=1.0,
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0,
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0,
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=30.5,
            step=0.1,
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
        )

    submitted = st.form_submit_button(
        "Predict Risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    patient_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    try:
        with st.spinner("Predicting risk..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=patient_data,
                timeout=30,
            )

        response.raise_for_status()
        result = response.json()
        prediction = result["prediction"]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "High Risk":
            st.error(f"Predicted category: {prediction}")
        elif prediction == "Low Risk":
            st.success(f"Predicted category: {prediction}")
        else:
            st.warning(f"Predicted category: {prediction}")


    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the prediction API. "
            "Check that the FastAPI container is running."
        )

    except requests.exceptions.Timeout:
        st.error("The API request timed out. Please try again.")

    except requests.exceptions.HTTPError as exc:
        st.error(f"The API returned an error: {exc}")

    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        st.error(f"Could not process the prediction: {exc}")