import streamlit as st
import requests
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Intent Classification",
    layout="wide"
)

API_BASE_URL = "http://localhost:8000/api"

st.title("Intent Classification System")
st.markdown("Classify user queries into predefined intent categories using a deployed ML model.")

# --- API Functions ---
def get_model_info():
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def classify_text(text):
    try:
        response = requests.post(f"{API_BASE_URL}/classify", json={"text": text})
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def classify_batch(texts):
    try:
        response = requests.post(f"{API_BASE_URL}/classify/batch", json={"texts": texts})
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# --- Sidebar: Model Info ---
with st.sidebar:
    st.header("Model Details")
    model_info = get_model_info()

    if model_info:
        st.success("API connected")
        st.markdown(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.markdown(f"**Number of Intents:** {len(model_info.get('classes', []))}")
        st.markdown("**Intents:**")
        for intent in model_info.get('classes', []):
            st.markdown(f"- {intent}")
    else:
        st.error("API not reachable")
        st.caption("Please ensure the FastAPI server is running on port 8000.")

# --- Tabs: Single and Batch ---
tab1, tab2 = st.tabs(["Single Query", "Batch Processing"])

# === Single Query Tab ===
with tab1:
    st.subheader("Single Query Classification")

    user_input = st.text_area("Input query", height=100, placeholder="E.g., Schedule a meeting with the team.")

    classify_btn = st.button("Classify")

    if classify_btn:
        if user_input.strip():
            with st.spinner("Processing..."):
                result = classify_text(user_input)
            if result:
                st.markdown("#### Prediction Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Intent", value=result["intent"])
                with col2:
                    st.metric(label="Confidence", value=f"{result['confidence']:.2%}")
            else:
                st.error("Failed to classify. Check API status.")
        else:
            st.warning("Please enter a query to classify.")

# === Batch Query Tab ===
with tab2:
    st.subheader("Batch Query Classification")

    batch_input = st.text_area(
        "Input multiple queries (one per line):",
        height=150,
        placeholder="E.g.,\nSend an email to the HR team\nWhat's the weather like tomorrow?"
    )

    batch_btn = st.button("Classify Batch")

    if batch_btn:
        if batch_input.strip():
            queries = [line.strip() for line in batch_input.split('\n') if line.strip()]
            with st.spinner(f"Classifying {len(queries)} queries..."):
                result = classify_batch(queries)

            if result and "results" in result:
                df = pd.DataFrame(result["results"])
                df["confidence"] = df["confidence"].apply(lambda x: f"{x:.2%}")
                
                st.markdown("#### Classification Results")
                st.dataframe(df, use_container_width=True)

                # Summary
                original_df = pd.DataFrame(result["results"])
                st.markdown("#### Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Queries", len(queries))
                with col2:
                    st.metric("Average Confidence", f"{original_df['confidence'].mean():.2%}")
                with col3:
                    most_common = original_df['intent'].value_counts().idxmax()
                    st.metric("Most Common Intent", most_common)
            else:
                st.error("Failed to process batch. Check API status.")
        else:
            st.warning("Please input some queries to process.")

# --- Footer ---
st.markdown("---")
st.caption("Powered by FastAPI and Streamlit · Developed for professional NLP use cases")
