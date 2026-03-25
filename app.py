import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Machine Failure Prediction Dashboard")

# Upload new machine data
uploaded_file = st.file_uploader("Upload new machine data (CSV)", type="csv")

if uploaded_file is not None:
    # Load uploaded data
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Preprocessing
    X_new = df.drop(['UDI', 'Product ID', 'Machine failure'], axis=1, errors='ignore')
    X_new = pd.get_dummies(X_new, columns=['Type'], drop_first=True)

    # Load saved Random Forest model
    rf_model = joblib.load("rf_model.pkl")

    # Automatically add any missing dummy columns from training
    for col in getattr(rf_model, "feature_names_in_", []):
        if col not in X_new.columns:
            X_new[col] = 0
    # Ensure column order matches model
    if hasattr(rf_model, "feature_names_in_"):
        X_new = X_new[rf_model.feature_names_in_]

    # Prediction
    predictions = rf_model.predict(X_new)
    df['Predicted_Failure'] = predictions

    # Show first 5 rows of predictions
    st.subheader("First 5 Predictions")
    st.dataframe(df[['UDI', 'Predicted_Failure']].head())

    # Show summary chart
    st.subheader("Predicted Failure Summary")
    failure_counts = df['Predicted_Failure'].value_counts()
    st.bar_chart(failure_counts)

    # Download predictions
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predicted_failures.csv',
        mime='text/csv',
    )