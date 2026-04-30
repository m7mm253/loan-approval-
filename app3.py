import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Credit Risk AI Dashboard", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- APP HEADER ---
st.title("📊 Financial Credit Scoring System")
st.subheader("AI-Driven Risk Assessment & Profitability Analysis")
st.markdown("---")

# --- SIDEBAR: INPUT FEATURES (5Cs of Credit) ---
st.sidebar.header("👤 Customer Profile Input")

def get_user_inputs():
    # Capacity & Capital
    income = st.sidebar.number_input("Monthly Income (EGP)", min_value=0, value=15000, step=500)
    loan_amount = st.sidebar.number_input("Requested Loan Amount", min_value=0, value=50000, step=1000)
    
    # Character
    credit_history = st.sidebar.selectbox(
        "Credit History Status", 
        options=["Excellent", "Good", "No History", "Previous Default"]
    )
    
    # Conditions & Demographics
    age = st.sidebar.slider("Customer Age", 21, 65, 35)
    employment_years = st.sidebar.slider("Years of Employment", 0, 40, 5)
    
    # Data structure for the model
    data = {
        'Monthly_Income': income,
        'Loan_Amount': loan_amount,
        'Age': age,
        'Employment_Years': employment_years,
        'Credit_Score_Encoded': 1 if credit_history in ["Excellent", "Good"] else 0
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

# --- MAIN INTERFACE LAYOUT ---
col_data, col_res = st.columns([1, 1])

with col_data:
    st.info("### 📝 Input Summary")
    st.write(input_df)

# --- AI PREDICTION LOGIC ---
# In a real scenario, you would use: model = joblib.load('your_model.pkl')
# For this demo, we use a logic-based simulation:
income_val = input_df['Monthly_Income'][0]
loan_val = input_df['Loan_Amount'][0]

# Simulated Probability (Lower is better)
base_prob = 0.10
if income_val < (loan_val * 0.1): base_prob += 0.40 # Low income vs loan ratio
if input_df['Credit_Score_Encoded'][0] == 0: base_prob += 0.30 # Poor history

probability = min(base_prob, 0.99)
prediction = 1 if probability > 0.50 else 0

with col_res:
    st.info("### 🤖 Model Prediction")
    if prediction == 0:
        st.success(f"**Status: APPROVED**")
        st.write(f"Default Probability: **{probability:.2%}**")
    else:
        st.error(f"**Status: REJECTED**")
        st.write(f"Default Probability: **{probability:.2%}**")

# --- COST ANALYST SECTION: PROFITABILITY ---
st.markdown("---")
st.header("💰 Financial Impact Analysis")

# Calculations
interest_rate = 0.22 # 22% Interest
potential_revenue = loan_val * interest_rate
expected_loss = loan_val * probability # Expected Loss (EL) = PD * EAD
net_profit = potential_revenue - expected_loss

m1, m2, m3 = st.columns(3)
m1.metric("Potential Revenue", f"{potential_revenue:,.2f} EGP")
m2.metric("Expected Risk Cost", f"-{expected_loss:,.2f} EGP", delta_color="inverse")
m3.metric("Projected Net Profit", f"{net_profit:,.2f} EGP")

# --- VISUALIZATIONS ---
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Key Decision Factors")
    # Simulation of Feature Importance
    importance_data = pd.DataFrame({
        'Feature': ['Credit History', 'Income Level', 'Employment', 'Loan Amount'],
        'Importance': [0.4, 0.3, 0.2, 0.1]
    })
    st.bar_chart(importance_data.set_index('Feature'))

with c2:
    st.subheader("Risk vs. Reward")
    # Simple chart showing Revenue vs Risk
    chart_data = pd.DataFrame({
        'Category': ['Revenue', 'Risk Cost'],
        'Amount': [potential_revenue, expected_loss]
    })
    st.vega_lite_chart(chart_data, {
        'mark': {'type': 'arc', 'innerRadius': 50},
        'encoding': {
            'theta': {'field': 'Amount', 'type': 'quantitative'},
            'color': {'field': 'Category', 'type': 'nominal'},
        }
    })

st.caption("Disclaimer: This is a decision-support tool. Final credit approval depends on internal policy compliance.")