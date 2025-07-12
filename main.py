import streamlit as st
import pandas as pd
from io import BytesIO

from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils2.main_utils import load_object
#Akshit's app
st.title("Sensor Fault Prediction App")

# 🧠 TRAINING SECTION
st.header("🔧 Model Training")
if st.button("Train Model"):
    try:
        pipeline = TrainPipeline()
        if pipeline.is_pipeline_running:
            st.warning("Training pipeline is already running.")
        else:
            pipeline.run_pipeline()
            st.success("Training pipeline completed successfully!")
    except Exception as e:
        st.error(f"Training failed: {e}")

# 🔍 PREDICTION SECTION
st.header("📁 Upload File for Prediction")

@st.cache_resource
def get_model():
    model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
    if not model_resolver.is_model_exists():
        return None, "No trained model found."
    
    model_path = model_resolver.get_best_model_path()
    model = load_object(file_path=model_path)
    return model, None

uploaded_file = st.file_uploader("Upload a preprocessed CSV file for prediction", type=["csv", "CSV"])

if uploaded_file is not None:
    try:
        st.success(f"Uploaded file: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())
        st.write("CSV Columns:", df.columns.tolist())

        if "class" in df.columns:
            df = df.drop(columns=["class"])

        model, model_error = get_model()
        if model_error:
            st.warning(model_error)
        else:
            try:
                expected_cols = list(model.model.feature_names_in_)  # Try accessing internal sklearn model
            except AttributeError:
                expected_cols = df.columns.tolist()
                st.warning("Model does not expose feature list — using uploaded columns as fallback.")

            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                df = df[expected_cols]
                predictions = model.predict(df)
                df["predicted"] = predictions
                df["predicted"].replace(TargetValueMapping().reverse_mapping(), inplace=True)

                st.subheader("Prediction Results")
                st.dataframe(df[["predicted"]])

                def convert_df_to_csv_bytes(df):
                    return BytesIO(df.to_csv(index=False).encode("utf-8"))

                csv_bytes = convert_df_to_csv_bytes(df)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error during processing: {e}")
else:
    st.info("Please upload a CSV file to begin.")
