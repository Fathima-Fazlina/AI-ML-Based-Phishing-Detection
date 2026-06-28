import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.url_detector import predict as predict_url, extract_features as url_features
from src.email_detector import predict as predict_email

st.set_page_config(page_title="Phishing Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ AI Phishing Detector")
st.caption("Detects phishing in URLs and emails using ML")

tab1, tab2 = st.tabs(["🔗 URL Checker", "📧 Email Checker"])

with tab1:
    st.subheader("Check a URL")
    url_input = st.text_input("Paste a URL", placeholder="https://example.com")
    
    if st.button("Analyze URL", type="primary"):
        if url_input:
            with st.spinner("Analyzing..."):
                result = predict_url(url_input)
                prob = result['phishing_probability']

            col1, col2 = st.columns(2)
            with col1:
                color = "🔴" if result['is_phishing'] else "🟢"
                verdict = "PHISHING" if result['is_phishing'] else "LEGITIMATE"
                st.metric("Verdict", f"{color} {verdict}")
                st.metric("Phishing probability", f"{prob:.1%}")

            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "crimson" if prob > 0.5 else "green"}},
                    title={'text': "Risk Score"}
                ))
                fig.update_layout(height=220, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Feature breakdown"):
                feats = url_features(url_input)
                st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "Value"}))

with tab2:
    st.subheader("Check an email")
    email_input = st.text_area("Paste raw email content", height=200,
        placeholder="Paste the full email text here...")

    if st.button("Analyze Email", type="primary"):
        if email_input:
            with st.spinner("Analyzing..."):
                result = predict_email(email_input)
                prob = result['phishing_probability']

            verdict = "🔴 PHISHING" if result['is_phishing'] else "🟢 LEGITIMATE"
            st.metric("Verdict", verdict)
            st.metric("Phishing probability", f"{prob:.1%}")
            st.caption("Preview: " + result['body_preview'])
