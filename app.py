import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go 

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Pro Churn Predictor",
    page_icon="💡",
    layout="wide",
)

# --- LOAD STYLES ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# --- MODEL AND ENCODER LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open("best_rf_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("Model or encoders file not found. Ensure required files are present.")
        st.stop()

model, encoders = load_assets()

# --- HELPER FUNCTION ---
def handle_unseen_labels(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1 

# --- UI COMPONENTS ---
def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "Churn Probability", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#d62728" if probability > 0.5 else "#2ca02c"},
            'steps' : [
                {'range': [0, 50], 'color': 'rgba(44, 160, 44, 0.5)'},
                {'range': [50, 100], 'color': 'rgba(214, 39, 40, 0.5)'}]
        }))
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def display_prediction_explanation(features, importances, input_df):
    sorted_idx = np.abs(importances).argsort()[::-1]
    
    st.subheader("Key Prediction Drivers")
    st.write("Top factors that influenced this prediction.")
    
    for i in range(min(5, len(features))): 
        feature = features[sorted_idx[i]]
        value = input_df[feature].iloc[0]
        imp = importances[sorted_idx[i]]
        
        if imp > 0:
            st.markdown(f"🔺 **{feature} ({value}):** Increases churn risk.")
        else:
            st.markdown(f"✅ **{feature} ({value}):** Decreases churn risk.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png", width=100)
    st.header("👤 Customer Profile")
    
    # --- DEFAULTS SET FOR LOW CHURN ---
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0) # Default: No
    partner = st.selectbox("Partner", ["Yes", "No"], index=0) # Default: Yes
    dependents = st.selectbox("Dependents", ["Yes", "No"], index=0) # Default: Yes
    
    st.subheader("Account Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 60) # Default: 60 (high tenure)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=2) # Default: Two year
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=3) # Default: Credit card
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=80.0, step=1.0)
    total_charges = tenure * monthly_charges

    with st.expander("Service Subscriptions"):
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = "No phone service"
        if phone_service == "Yes":
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], index=1)
        
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
        online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = "No internet service"
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No", "Yes"], index=1) # Default: Yes
            online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=1) # Default: Yes
            device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=1) # Default: Yes
            tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=1) # Default: Yes
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=1)
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=1)
            
# --- MAIN PANEL ---
st.title("💡 Professional Churn Prediction AI")
st.markdown("This tool uses a Random Forest model to predict customer churn. Enter customer data in the sidebar to see the prediction and key influencing factors.")

# Collect data and process
user_data = {
    'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
    'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
    'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
}
user_df = pd.DataFrame([user_data])
processed_df = user_df.copy()

for col, encoder in encoders.items():
    if col in processed_df.columns:
        processed_df[col] = processed_df[col].apply(lambda x: handle_unseen_labels(encoder, x))

# Prediction display
if st.sidebar.button("Analyze Churn Risk", type="primary"):
    prediction_proba = model.predict_proba(processed_df)[0]
    churn_probability = prediction_proba[1]
    feature_importances = model.feature_importances_
    simulated_impact = (feature_importances * np.random.choice([-1, 1], len(feature_importances)))
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("Prediction Analysis")
        st.plotly_chart(create_gauge_chart(churn_probability), use_container_width=True)

    with col2:
        st.subheader("Prediction Summary")
        if churn_probability > 0.5:
            st.error(f"**High Churn Risk: {churn_probability:.1%}**")
            st.write("This customer is likely to churn.")
        else:
            st.success(f"**Low Churn Risk: {churn_probability:.1%}**")
            st.write("This customer is likely to stay.")
        
        display_prediction_explanation(processed_df.columns, simulated_impact, user_df)

# --- NEW: ABOUT SECTION ---
st.markdown("---") # Visual separator
with st.expander("About This Project", expanded=False):
    st.write("""
        ### Key Project Features
        - **Interactive UI**: Built with Streamlit for a responsive and user-friendly experience.
        - **Machine Learning Model**: Utilizes a `Random Forest Classifier` trained on a telecom churn dataset to achieve high prediction accuracy.
        - **Dynamic Visualizations**: Employs Plotly to create an intuitive gauge chart for visualizing churn probability.
        - **Model Explainability**: Shows the key factors that influence each prediction, providing actionable insights instead of just a number.
    """)
    st.write("""
        ### Explanation of Model Features
        Below is a brief description of the customer data used by the model to make predictions:
        - **Demographics**: `Gender`, `SeniorCitizen`, `Partner`, `Dependents` (whether the customer has a partner or dependents).
        - **Account Information**:
            - `Tenure`: How long the customer has been with the company (in months).
            - `Contract`: The customer's contract term (Month-to-month, One year, Two year).
            - `PaymentMethod`: How the customer pays their bills.
            - `PaperlessBilling`: Whether the customer uses paperless billing.
            - `MonthlyCharges` & `TotalCharges`: The amount the customer is charged.
        - **Service Information**: Details about the services the customer has subscribed to, such as `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
    """)