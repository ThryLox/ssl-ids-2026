import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import src.models_contrastive as models_contrastive
import src.score as score

st.set_page_config(page_title="Intrusion Detection SOC", layout="wide")

st.title("🛡️ Contrastive SSL Intrusion Detection Dashboard")
st.markdown("""
This dashboard identifies anomalies in network traffic using self-supervised representations. 
It utilizes a Mahalanobis distance metric in a contrastive latent space to detect unseen attack patterns.
""")

@st.cache_resource
def load_model_and_scaler():
    # Load data for scaler fitting (using a small sample)
    df = pd.read_csv('results/cleaned_sample_full.csv', nrows=10000)
    df_numeric = df.select_dtypes(include=[np.number])
    scaler = StandardScaler().fit(df_numeric)
    
    input_dim = df_numeric.shape[1]
    model = models_contrastive.ContrastiveEncoder(input_dim=input_dim)
    # Note: We assume results/contrastive_encoder.pth exists
    if os.path.exists("results/contrastive_encoder.pth"):
        model.load_state_dict(torch.load("results/contrastive_encoder.pth"))
    model.eval()
    
    # Fit Mahalanobis on normal train data (placeholder or cached)
    # For prototype, we'll re-calculate on a small normal slice
    normal_slice = df[df['Label'].astype(str).str.contains('BENIGN', na=False)].select_dtypes(include=[np.number])
    if len(normal_slice) == 0:
        # Fallback if BENIGN not in first 10k
        normal_slice = df_numeric.head(100)
    
    normal_scaled = scaler.transform(normal_slice)
    
    with torch.no_grad():
        h, _ = model(torch.FloatTensor(normal_scaled))
    mu, inv_cov = score.fit_gaussian_embeddings(h.numpy())
    
    return model, scaler, mu, inv_cov

model, scaler, mu, inv_cov = load_model_and_scaler()

st.sidebar.header("Upload Traffic Logs")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df_data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df_data.head())
    
    if st.button("🚀 Analyze Traffic"):
        with st.spinner("Decoding representations..."):
            numeric_data = df_data.select_dtypes(include=[np.number])
            # Ensure columns match
            X_scaled = scaler.transform(numeric_data)
            
            with torch.no_grad():
                h, _ = model(torch.FloatTensor(X_scaled))
            
            scores = score.score_mahalanobis(h.numpy(), mu, inv_cov)
            
            # Calibration - simplistic threshold for prototype
            threshold = np.percentile(scores, 95) 
            
            df_data['AnomalyScore'] = scores
            df_data['Level'] = df_data['AnomalyScore'].apply(lambda x: "High" if x > threshold*1.5 else "Medium" if x > threshold else "Low")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Flows", len(df_data))
            with col2:
                st.metric("Critical Alerts", len(df_data[df_data['Level'] == "High"]))
            
            st.subheader("Anomaly Score Trend")
            st.line_chart(df_data['AnomalyScore'])
            
            st.subheader("Security Alerts")
            alerts = df_data[df_data['Level'] != "Low"].sort_values(by='AnomalyScore', ascending=False)
            st.dataframe(alerts)
            
            if not alerts.empty:
                st.error(f"Potential Intrusion Detected! {len(alerts)} suspicious flows found.")
else:
    st.info("👈 Upload a network flow CSV (e.g., from CIC-IDS2017) to begin analysis.")

    # Show some pre-generated graphs
    st.divider()
    st.subheader("Pre-trained Model Performance")
    colA, colB = st.columns(2)
    with colA:
        st.image("results/roc_DoS.png", caption="ROC Curve (DoS)")
    with colB:
        st.image("results/tsne_DoS.png", caption="Latent Space t-SNE")
