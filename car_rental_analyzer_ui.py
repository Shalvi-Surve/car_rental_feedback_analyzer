import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import numpy as np
from transformers import pipeline
from collections import Counter
import streamlit as st
import plotly.express as px
import plotly
import time
import logging
import openpyxl

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Streamlit page configuration
st.set_page_config(page_title="Car Rental Feedback Analyzer", layout="wide")

# Custom CSS for Neon Cyberpunk (Enhanced)
st.markdown("""
<style>
body {
    background: linear-gradient(to bottom, #03071e, #370617);
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 100 100"><rect width="100%" height="100%" fill="none"/><g><path d="M0 0 L100 100" stroke="#00f5d4" stroke-width="0.3" opacity="0.3"/><path d="M100 0 L0 100" stroke="#f72585" stroke-width="0.3" opacity="0.3"/><animate attributeName="opacity" values="0.3;0.6;0.3" dur="3s" repeatCount="indefinite"/></g></svg>') no-repeat center center fixed;
    background-size: cover;
}
h1, h2, h3 {
    color: #00f5d4;
    font-family: 'Montserrat', sans-serif;
    text-shadow: 0 0 12px #00f5d4, 0 0 24px #f72585;
    animation: glow 2s ease-in-out infinite;
}
@keyframes glow {
    0% { text-shadow: 0 0 12px #00f5d4, 0 0 24px #f72585; }
    50% { text-shadow: 0 0 18px #00f5d4, 0 0 30px #f72585; }
    100% { text-shadow: 0 0 12px #00f5d4, 0 0 24px #f72585; }
}
.hero {
    background: linear-gradient(45deg, #00f5d4, #f72585);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 25px #00f5d4;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 25px #00f5d4; }
    50% { box-shadow: 0 0 35px #f72585; }
    100% { box-shadow: 0 0 25px #00f5d4; }
}
.stButton>button {
    background: linear-gradient(45deg, #f72585, #7209b7);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 0 15px #f72585;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #7209b7, #f72585);
    box-shadow: 0 0 20px #00f5d4;
    transform: scale(1.05);
}
.stTextArea textarea {
    background-color: #ffffff;
    color: #1e1e1e;
    border: 2px solid #00f5d4;
    border-radius: 10px;
    font-size: 16px;
    padding: 12px;
}
.stSelectbox, .stSlider, .stNumberInput {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
    border: 2px solid #00f5d4;
}
.card {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 20px rgba(0, 245, 212, 0.4);
    transition: transform 0.3s ease;
    animation: slideIn 1s ease-in;
}
@keyframes slideIn {
    0% { transform: translateX(-100px); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}
.card:hover {
    transform: translateY(-8px);
}
.kpi-card {
    background: linear-gradient(45deg, #f72585, #7209b7);
    color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
    text-align: center;
    box-shadow: 0 0 15px #00f5d4;
    cursor: pointer;
    transition: transform 0.3s ease;
    animation: kpi-pulse 2s infinite;
}
@keyframes kpi-pulse {
    0% { box-shadow: 0 0 15px #00f5d4; }
    50% { box-shadow: 0 0 25px #f72585; }
    100% { box-shadow: 0 0 15px #00f5d4; }
}
.kpi-card:hover {
    transform: scale(1.05);
}
.metric-box {
    background: linear-gradient(45deg, #00f5d4, #f72585);
    color: #ffffff;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
    text-align: center;
    box-shadow: 0 0 15px #7209b7;
    animation: metric-pulse 2s infinite;
}
@keyframes metric-pulse {
    0% { box-shadow: 0 0 15px #7209b7; }
    50% { box-shadow: 0 0 20px #00f5d4; }
    100% { box-shadow: 0 0 15px #7209b7; }
}
.carousel {
    display: flex;
    overflow-x: auto;
    padding: 10px;
    gap: 20px;
}
.carousel-item {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    padding: 15px;
    min-width: 300px;
    box-shadow: 0 0 15px #f72585;
    transition: transform 0.3s ease;
}
.carousel-item:hover {
    transform: scale(1.05);
}
.plotly-chart {
    border: 2px solid #00f5d4;
    border-radius: 15px;
    background-color: #ffffff;
    padding: 20px;
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
.stExpander {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    border: 1px solid #00f5d4;
}
.stExpander summary {
    color: #00f5d4;
    font-weight: bold;
}
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #f72585;
    color: #ffffff;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
}
.summary-table {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0, 245, 212, 0.4);
}
.summary-table th {
    color: #f72585;
    font-weight: bold;
}
.summary-table td {
    color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)

# JavaScript for enhanced confetti
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
<script>
function triggerConfetti() {
    confetti({
        particleCount: 150,
        spread: 90,
        origin: { y: 0.6 },
        colors: ['#00f5d4', '#f72585', '#7209b7']
    });
}
</script>
""", unsafe_allow_html=True)

# Step 1: Load data
@st.cache_data
def load_data():
    logging.info("Step 1: Loading datasets...")
    start_time = time.time()
    try:
        train_data = pd.read_csv("car_rental_training_data (1).csv")
        test_data = pd.read_csv("car_rental_test_data (1).csv")
        logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    except FileNotFoundError as e:
        logging.error(f"CSV file not found: {e}")
        st.error("Error: CSV files not found. Ensure 'car_rental_training_data (1).csv' and 'car_rental_test_data (1).csv' are in the project directory.")
        return None, None
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        st.error(f"Error loading CSV: {e}")
        return None, None
    logging.info(f"Dataset loading completed in {time.time() - start_time:.2f} seconds")
    return train_data, test_data

# Step 2: Clean data
@st.cache_data
def clean_data(df):
    start_time = time.time()
    logging.info("Step 2: Cleaning data...")
    df = df.dropna(subset=["Customer_Service", "Satisfaction", "Business_Area"])
    df["Customer_Service"] = df["Customer_Service"].str.strip().replace(r'^"|"$' , '', regex=True)
    df["Satisfaction"] = df["Satisfaction"].astype(str)
    logging.info(f"Cleaned data shape: {df.shape}")
    logging.info(f"Data cleaning completed in {time.time() - start_time:.2f} seconds")
    return df

# Step 3: Initialize classifiers
@st.cache_resource
def init_classifiers():
    logging.info("Step 3: Initializing transformer-based classifiers...")
    start_time = time.time()
    try:
        sentiment_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
        business_area_classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", framework="pt")
    except Exception as e:
        logging.error(f"Error initializing classifiers: {e}")
        st.error(f"Error initializing classifiers: {e}")
        return None, None
    logging.info(f"Classifier initialization completed in {time.time() - start_time:.2f} seconds")
    return sentiment_classifier, business_area_classifier

# Step 4: Sentiment and business area classification
def classify_sentiment(comments, classifier):
    try:
        prompt = [f"Was customer satisfied?\ncomment: {comment}\n" for comment in comments]
        results = classifier(prompt, batch_size=8)
        return ["satisfied" if r["label"] == "POSITIVE" else "dissatisfied" for r in results]
    except Exception as e:
        logging.error(f"Error in sentiment classification: {e}")
        return ["unknown"] * len(comments)

def classify_business_area(comments, classifier):
    business_areas = [
        'Product: Functioning', 'Product: Pricing and Billing', 'Service: Accessibility',
        'Service: Attitude', 'Service: Knowledge', 'Service: Orders/Contracts'
    ]
    try:
        prompt = [f"Find the business area of the customer comment.\nChoose business area from: {', '.join(business_areas)}.\ncomment: {comment}\n" for comment in comments]
        results = classifier(prompt, candidate_labels=business_areas, batch_size=8)
        return [r["labels"][0] for r in results]
    except Exception as e:
        logging.error(f"Error in business area classification: {e}")
        return ["unknown"] * len(comments)

# Step 5: Precompute results for efficiency
@st.cache_data
def compute_results(_sentiment_classifier, _business_area_classifier, test_data, max_comments=100):
    logging.info(f"Step 4: Precomputing test data results (max {max_comments} comments)...")
    start_time = time.time()
    comments = test_data["Customer_Service"].head(max_comments).tolist()
    progress_bar = st.progress(0)
    results_sentiment = []
    results_business_area = []
    batch_size = 8
    for i in range(0, len(comments), batch_size):
        batch_comments = comments[i:i + batch_size]
        results_sentiment.extend(classify_sentiment(batch_comments, _sentiment_classifier))
        results_business_area.extend(classify_business_area(batch_comments, _business_area_classifier))
        progress_bar.progress(min((i + batch_size) / len(comments), 1.0))
    logging.info(f"Test data processing completed in {time.time() - start_time:.2f} seconds")
    return results_sentiment, results_business_area

# Main Streamlit app
def main():
    # Hero section
    st.markdown("""
    <div class='hero'>
        <h1>🚗 Car Rental Feedback Analyzer</h1>
        <p style='color: #ffffff; font-size: 20px; font-weight: bold;'>Unlock Customer Insights with AI-Powered Cyberpunk Dashboards</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation and settings
    with st.sidebar:
        st.markdown("<h2 style='color: #f72585;'>Dashboard</h2>", unsafe_allow_html=True)
        tab_selection = st.radio("", ["Overview", "Dataset", "Analyze Feedback", "Key Findings", "Insights"], label_visibility="collapsed")
        st.markdown("<hr style='border-color: #00f5d4;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #00f5d4;'>Settings</h3>", unsafe_allow_html=True)
        max_comments = st.slider("Comments per page:", 5, 50, 10, help="Adjust to control comment display speed")
        max_predictions = st.slider("Comments to analyze:", 10, 500, 100, step=10, help="Lower for faster loading")
        st.markdown("<p style='color: #ffffff;'>Tune for cyberpunk vibes! ✨</p>", unsafe_allow_html=True)

    # Load data
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    if test_data.empty:
        st.error("Test dataset is empty. Please check 'car_rental_test_data (1).csv'.")
        return

    # Clean data
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    if test_data.empty:
        st.error("Test dataset is empty after cleaning. Please verify the data.")
        return

    # Initialize classifiers
    sentiment_classifier, business_area_classifier = init_classifiers()
    if sentiment_classifier is None or business_area_classifier is None:
        return

    # Precompute results
    results_sentiment, results_business_area = compute_results(sentiment_classifier, business_area_classifier, test_data, max_comments=max_predictions)

    # Tab 1: Overview
    if tab_selection == "Overview":
        st.markdown("<div class='card'><h2>🌟 Project Overview</h2>", unsafe_allow_html=True)
        with st.expander("About This Project", expanded=True):
            st.markdown("""
            <p style='color: #1e1e1e;'>The <b>Car Rental Feedback Analyzer</b> is a cutting-edge AI-powered dashboard built for the IBM Gen AI Summer Course (July 2025). It leverages NLP to analyze customer feedback, delivering actionable insights for car rental businesses.</p>
            """, unsafe_allow_html=True)
        with st.expander("Key Features"):
            st.markdown("""
            <ul style='color: #1e1e1e;'>
                <li>🔍 <b>Sentiment Analysis</b>: Classifies feedback as satisfied or dissatisfied using DistilBERT.</li>
                <li>🏷️ <b>Business Area Detection</b>: Pinpoints areas like Service: Attitude or Product: Pricing.</li>
                <li>📊 <b>Cyberpunk Dashboards</b>: Interactive charts and KPIs for trends.</li>
                <li>✍️ <b>Real-Time Insights</b>: Analyze custom comments instantly.</li>
                <li>💾 <b>Data Export</b>: Download results as Excel for reporting.</li>
            </ul>
            """, unsafe_allow_html=True)
        with st.expander("Tech Stack"):
            st.markdown("""
            <p style='color: #1e1e1e;'>Python, Streamlit, Pandas, Transformers (Hugging Face), Plotly, Openpyxl</p>
            """, unsafe_allow_html=True)
        with st.expander("Purpose"):
            st.markdown("""
            <p style='color: #1e1e1e;'>Empowers car rental companies to enhance services and showcases AI/data visualization skills for portfolios.</p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Dataset
    elif tab_selection == "Dataset":
        st.markdown("<div class='card'><h2>📊 Dataset Insights</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='tooltip'><h3>Training Data</h3><span class='tooltiptext'>Summary of training dataset</span></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #1e1e1e;'>{train_data.shape[0]} rows, {train_data.shape[1]} columns</p>", unsafe_allow_html=True)
            st.dataframe(train_data['Satisfaction'].value_counts(), use_container_width=True)
            try:
                with open("C:/Users/Shalvi/OneDrive/Desktop/IBM GEN AI/PROJECT/CAR RENTAL CUSTOMER FEEDBACK ANALYSER/car_rental_training_data (1).csv", "rb") as file:
                    st.download_button("Download Training Data CSV", file, "car_rental_training_data.csv", key="download_train")
                    st.markdown("<script>triggerConfetti();</script>", unsafe_allow_html=True)
            except FileNotFoundError:
                st.error("Training data CSV not found.")
        with col2:
            st.markdown("<div class='tooltip'><h3>Test Data</h3><span class='tooltiptext'>Summary of test dataset</span></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #1e1e1e;'>{test_data.shape[0]} rows, {test_data.shape[1]} columns</p>", unsafe_allow_html=True)
            st.dataframe(test_data['Satisfaction'].value_counts(), use_container_width=True)
            try:
                with open("C:/Users/Shalvi/OneDrive/Desktop/IBM GEN AI/PROJECT/CAR RENTAL CUSTOMER FEEDBACK ANALYSER/car_rental_test_data (1).csv", "rb") as file:
                    st.download_button("Download Test Data CSV", file, "car_rental_test_data.csv", key="download_test")
                    st.markdown("<script>triggerConfetti();</script>", unsafe_allow_html=True)
            except FileNotFoundError:
                st.error("Test data CSV not found.")
        with st.expander("Sample Training Comments"):
            page = st.number_input("Page:", min_value=1, max_value=(len(train_data) // max_comments) + 1, value=1, step=1)
            start_idx = (page - 1) * max_comments
            end_idx = start_idx + max_comments
            for idx, row in train_data.iloc[start_idx:end_idx].iterrows():
                st.markdown(f"<div class='card'><p style='color: #1e1e1e;'><b>Comment {idx+1}</b>: {row['Customer_Service'][:100]}...</p><p style='color: #1e1e1e;'>Satisfaction: {row['Satisfaction']}, Business Area: {row['Business_Area']}</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: Analyze Feedback
    elif tab_selection == "Analyze Feedback":
        st.markdown("<div class='card'><h2>✍️ Real-Time Feedback Analysis</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_area("Enter a customer comment:", height=150, placeholder="e.g., 'The car was clean but the staff was rude.'")
            if st.button("Analyze Comment", key="analyze"):
                if user_input:
                    sentiment = classify_sentiment([user_input], sentiment_classifier)[0]
                    business_area = classify_business_area([user_input], business_area_classifier)[0]
                    st.markdown(f"<div class='card'><h3 style='color: #f72585;'>Results</h3><p style='color: #1e1e1e;'><b>Sentiment</b>: {sentiment.capitalize()}</p><p style='color: #1e1e1e;'><b>Business Area</b>: {business_area}</p></div>", unsafe_allow_html=True)
                    st.markdown("<script>triggerConfetti();</script>", unsafe_allow_html=True)
                else:
                    st.warning("Please enter a comment to analyze.")
        with col2:
            with st.expander("Quick Stats"):
                st.markdown(f"<p style='color: #1e1e1e;'><b>Total Comments Analyzed</b>: {len(test_data)}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #1e1e1e;'><b>Satisfaction Rate</b>: {(test_data['Satisfaction'] == '1').mean() * 100:.2f}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 4: Key Findings (Replaces Results)
    elif tab_selection == "Key Findings":
        st.markdown("<div class='card'><h2>🔍 Key Findings</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #1e1e1e;'>Explore AI-driven insights with interactive metrics and a sleek carousel.</p>", unsafe_allow_html=True)

        # Prepare data
        total_comments = len(results_sentiment)
        sentiment_counts = Counter(results_sentiment)
        satisfaction_rate = (sentiment_counts["satisfied"] / total_comments) * 100 if total_comments > 0 else 0
        business_area_counts = Counter(results_business_area)
        business_area_summary = {area: count / total_comments * 100 for area, count in business_area_counts.items() if total_comments > 0}
        frequent_issues = test_data[test_data["Satisfaction"] == "0"]["Customer_Service"].value_counts().head(5)

        # Key Metrics
        st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-box'><h3>Total Comments</h3><p style='font-size: 24px;'>{total_comments}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><h3>Satisfaction Rate</h3><p style='font-size: 24px;'>{satisfaction_rate:.2f}%</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-box'><h3>Satisfied Comments</h3><p style='font-size: 24px;'>{sentiment_counts.get('satisfied', 0)}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'><h3>Dissatisfied Comments</h3><p style='font-size: 24px;'>{sentiment_counts.get('dissatisfied', 0)}</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Top 5 Frequent Issues Carousel
        st.markdown("<div class='tooltip'><h3>Top 5 Frequent Issues (Dissatisfied)</h3><span class='tooltiptext'>Scroll to view common customer complaints</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='carousel'>", unsafe_allow_html=True)
        for issue, count in frequent_issues.items():
            st.markdown(f"<div class='carousel-item'><p style='color: #1e1e1e;'>{issue[:100]}...: {count} occurrences</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Interactive Predictions Table
        st.markdown("<div class='tooltip'><h3>Sample Predictions</h3><span class='tooltiptext'>Search and filter predictions</span></div>", unsafe_allow_html=True)
        results_df = pd.DataFrame({
            "Comment": test_data["Customer_Service"].head(max_predictions),
            "Predicted_Sentiment": results_sentiment,
            "Predicted_Business_Area": results_business_area,
            "Actual_Sentiment": test_data["Satisfaction"].head(max_predictions),
            "Actual_Business_Area": test_data["Business_Area"].head(max_predictions)
        })
        search_term = st.text_input("Search Comments:", placeholder="Enter keyword...")
        sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Satisfied", "Dissatisfied"])
        filtered_df = results_df
        if search_term:
            filtered_df = filtered_df[filtered_df["Comment"].str.contains(search_term, case=False, na=False)]
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df["Predicted_Sentiment"] == sentiment_filter.lower()]
        if not filtered_df.empty:
            page = st.number_input("Page:", min_value=1, max_value=(len(filtered_df) // max_comments) + 1, value=1, step=1, key="findings_page")
            start_idx = (page - 1) * max_comments
            end_idx = start_idx + max_comments
            st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
            if st.button("Apply Filters", key="apply_filters"):
                st.markdown("<script>triggerConfetti();</script>", unsafe_allow_html=True)
        else:
            st.warning("No results match the search criteria.")

        # Export Results as Multi-Sheet Excel
        st.markdown("<div class='tooltip'><h3>Export Results</h3><span class='tooltiptext'>Download analysis as Excel with multiple sheets</span></div>", unsafe_allow_html=True)
        logging.info("Step 6: Saving results to Excel...")
        start_time = time.time()
        output_path = "analysis_results.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Sheet 1: Big Table
            results_df.to_excel(writer, sheet_name="Full Results", index=False)
            # Sheet 2: Summary Table
            summary_data = {
                "Metric": [
                    "Total Comments",
                    "Satisfaction Rate",
                    "Top Business Area",
                    "Avg Comment Length",
                    "Satisfied Comments",
                    "Dissatisfied Comments"
                ],
                "Value": [
                    f"{total_comments}",
                    f"{satisfaction_rate:.2f}%",
                    max(business_area_counts.items(), key=lambda x: x[1])[0] if business_area_counts else "N/A",
                    f"{test_data['Customer_Service'].head(max_predictions).str.len().mean():.1f} chars",
                    f"{sentiment_counts.get('satisfied', 0)}",
                    f"{sentiment_counts.get('dissatisfied', 0)}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            # Sheet 3: Dashboard Data
            sentiment_data = pd.DataFrame({
                "Sentiment": list(sentiment_counts.keys()),
                "Count": list(sentiment_counts.values())
            })
            business_area_data = pd.DataFrame({
                "Business Area": list(business_area_summary.keys()),
                "Percentage": list(business_area_summary.values())
            })
            comment_length_bins = pd.cut(test_data["Customer_Service"].head(max_predictions).str.len(), bins=20).value_counts().reset_index()
            comment_length_bins.columns = ["Length Range", "Count"]
            dashboard_data = {
                "Sentiment Distribution": sentiment_data,
                "Business Area Distribution": business_area_data,
                "Comment Length Distribution": comment_length_bins
            }
            for name, df in dashboard_data.items():
                df.to_excel(writer, sheet_name="Dashboard Data", startrow=writer.sheets["Dashboard Data"].max_row + 2 if "Dashboard Data" in writer.sheets else 0, index=False)
        logging.info(f"Results saved to analysis_results.xlsx in {time.time() - start_time:.2f} seconds")
        with open(output_path, "rb") as file:
            st.download_button("Download Results Excel", file, "analysis_results.xlsx", key="download_excel")
            st.markdown("<script>triggerConfetti();</script>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 5: Insights
    elif tab_selection == "Insights":
        st.markdown("<div class='card'><h2>📊 Insights Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #1e1e1e;'>Power BI-style dashboard with real-time customer feedback insights.</p>", unsafe_allow_html=True)

        # Prepare dashboard data
        total_comments = len(results_sentiment)
        sentiment_counts = Counter(results_sentiment)
        satisfaction_rate = (sentiment_counts["satisfied"] / total_comments) * 100 if total_comments > 0 else 0
        business_area_counts = Counter(results_business_area)
        most_common_area = max(business_area_counts.items(), key=lambda x: x[1])[0] if business_area_counts else "N/A"
        comment_lengths = test_data["Customer_Service"].head(max_predictions).str.len()
        avg_comment_length = comment_lengths.mean() if not comment_lengths.empty else 0

        # KPI Cards
        st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='kpi-card' onclick='triggerConfetti()'><h3>Total Comments</h3><p style='font-size: 24px;'>{total_comments}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-card' onclick='triggerConfetti()'><h3>Satisfaction Rate</h3><p style='font-size: 24px;'>{satisfaction_rate:.2f}%</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='kpi-card' onclick='triggerConfetti()'><h3>Top Business Area</h3><p style='font-size: 20px;'>{most_common_area}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-card' onclick='triggerConfetti()'><h3>Avg Comment Length</h3><p style='font-size: 24px;'>{avg_comment_length:.1f} chars</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Interactive Charts
        st.markdown("<div class='grid-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='tooltip'><h3>Sentiment Distribution</h3><span class='tooltiptext'>Percentage of satisfied vs. dissatisfied comments</span></div>", unsafe_allow_html=True)
            try:
                if not sentiment_counts:
                    st.warning("No sentiment data to display.")
                else:
                    fig_sentiment = px.pie(
                        names=sentiment_counts.keys(),
                        values=sentiment_counts.values(),
                        title="Sentiment Distribution",
                        color_discrete_sequence=['#00f5d4', '#f72585'],
                        height=400,
                        hole=0.4
                    )
                    fig_sentiment.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hoverinfo='label+percent+value',
                        marker=dict(line=dict(color='#ffffff', width=2))
                    )
                    fig_sentiment.update_layout(
                        title_font_size=20,
                        font_size=14,
                        margin=dict(t=50, b=50),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering Sentiment Pie Chart: {e}")
                logging.error(f"Sentiment Pie Chart error: {e}")

        with col2:
            st.markdown("<div class='tooltip'><h3>Business Area Distribution</h3><span class='tooltiptext'>Percentage of comments per business area</span></div>", unsafe_allow_html=True)
            try:
                business_area_summary = {area: count / total_comments * 100 for area, count in business_area_counts.items()}
                if not business_area_summary:
                    st.warning("No business area data to display.")
                else:
                    fig_business = px.bar(
                        x=list(business_area_summary.keys()),
                        y=list(business_area_summary.values()),
                        title="Business Area Distribution",
                        color_discrete_sequence=['#00f5d4'],
                        height=400
                    )
                    fig_business.update_layout(
                        xaxis_title="Business Area",
                        yaxis_title="Percentage (%)",
                        xaxis_tickangle=45,
                        title_font_size=20,
                        font_size=14,
                        margin=dict(t=50, b=100),
                        showlegend=False
                    )
                    fig_business.update_traces(
                        hovertemplate='%{x}: %{y:.2f}%',
                        marker=dict(line=dict(color='#ffffff', width=2))
                    )
                    st.plotly_chart(fig_business, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering Business Area Bar Chart: {e}")
                logging.error(f"Business Area Bar Chart error: {e}")

        st.markdown("<div class='tooltip'><h3>Comment Length Distribution</h3><span class='tooltiptext'>Distribution of comment lengths in characters</span></div>", unsafe_allow_html=True)
        try:
            if comment_lengths.empty:
                st.warning("No comment length data to display.")
            else:
                fig_length = px.histogram(
                    x=comment_lengths,
                    nbins=20,
                    title="Comment Length Distribution",
                    color_discrete_sequence=['#f72585'],
                    height=400
                )
                fig_length.update_layout(
                    xaxis_title="Characters",
                    yaxis_title="Count",
                    title_font_size=20,
                    font_size=14,
                    margin=dict(t=50, b=50),
                    showlegend=False
                )
                fig_length.update_traces(
                    hovertemplate='Length: %{x}<br>Count: %{y}',
                    marker=dict(line=dict(color='#ffffff', width=2))
                )
                st.plotly_chart(fig_length, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering Comment Length Histogram: {e}")
            logging.error(f"Comment Length Histogram error: {e}")

        # Summary Table
        st.markdown("<div class='tooltip'><h3>Summary Table</h3><span class='tooltiptext'>Aggregated metrics for quick reference</span></div>", unsafe_allow_html=True)
        summary_data = {
            "Metric": [
                "Total Comments",
                "Satisfaction Rate",
                "Top Business Area",
                "Avg Comment Length",
                "Satisfied Comments",
                "Dissatisfied Comments"
            ],
            "Value": [
                f"{total_comments}",
                f"{satisfaction_rate:.2f}%",
                most_common_area,
                f"{avg_comment_length:.1f} chars",
                f"{sentiment_counts.get('satisfied', 0)}",
                f"{sentiment_counts.get('dissatisfied', 0)}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        st.markdown("<div class='summary-table'>", unsafe_allow_html=True)
        st.table(summary_df)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
