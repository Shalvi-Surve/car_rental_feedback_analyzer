# 🚗 Car Rental Feedback Analyzer

A vibrant, AI-powered Streamlit dashboard built for the **IBM Gen AI Summer Course (July 2025)** to analyze car rental customer feedback using NLP and deliver actionable insights with a Neon Cyberpunk UI.

## 🌟 Streamlit Badge
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://carrentalfeedbackanalyzer-ibm-shalvisurve.streamlit.app)

## 🌟 Demo
[Live Streamlit App](https://carrentalfeedbackanalyzer-ibm-shalvisurve.streamlit.app)

## 📖 Project Overview
This project leverages **DistilBERT** for sentiment analysis and zero-shot classification to categorize customer feedback into sentiments (satisfied/dissatisfied) and business areas (e.g., Service: Attitude, Product: Pricing). Built with **Streamlit** and **Plotly**, it features interactive dashboards, KPI cards, a searchable table, and exportable Excel reports, all styled with a futuristic Neon Cyberpunk aesthetic.

## 🔍 Key Features
- **Sentiment Analysis**: Classifies feedback as satisfied or dissatisfied using DistilBERT.
- **Business Area Detection**: Identifies areas like Service: Attitude or Product: Pricing.
- **Cyberpunk Dashboards**: Interactive charts (pie, bar, histogram) and KPI cards.
- **Real-Time Insights**: Analyze custom comments instantly.
- **Data Export**: Download results as multi-sheet Excel or CSV files with confetti effects.

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **Streamlit**: Web app framework for the dashboard.
- **Transformers (Hugging Face)**: For NLP models (`distilbert-base-uncased-finetuned-sst-2-english`, `typeform/distilbert-base-uncased-mnli`).
- **Pandas & NumPy**: Data processing.
- **Plotly**: Interactive visualizations.
- **Openpyxl**: Excel file generation.

## 📂 Project Structure
- `car_rental_analyzer_ui.py`: Main Streamlit app script.
- `car_rental_training_data (1).csv`: Training dataset (233 rows, 11 columns).
- `car_rental_test_data (1).csv`: Test dataset (162 rows, 11 columns).
- `requirements.txt`: Dependencies for deployment.
- `analysis_results.xlsx`: Sample output (generated dynamically, not in repo).

## 🚀 How to Run
**1. Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/car-rental-feedback-analyzer.git
   ```

**2. Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**3. Run the app locally:**
   ```bash
   streamlit run car_rental_analyzer_ui.py
   ```

**4. Access at:**
   http://localhost:8501.
   ```

## 🎯 Purpose
Empowers car rental companies to enhance services through AI-driven insights and showcases my skills in NLP, data visualization, and web development for portfolio and internship applications.


##🙌 Acknowledgments
IBM Gen AI Summer Course for the learning platform.
Hugging Face for pre-trained models.
Streamlit Community for deployment support.


Feel free to star ⭐ this repository if you find it useful!
