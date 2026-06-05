import streamlit as st
import pickle
import pandas as pd

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
with open("knn_neighbour.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
}

.subtitle {
    text-align:center;
    color:#dcdcdc;
    font-size:18px;
    margin-bottom:30px;
}

.card {
    background-color: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

.stButton>button {
    width:100%;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    font-size:20px;
    border:none;
    border-radius:10px;
    padding:12px;
}

.result-success {
    background:#28a745;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:28px;
    color:white;
    font-weight:bold;
}

.result-danger {
    background:#dc3545;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:28px;
    color:white;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="title">🛡️ Fraud Detection System</div>',
            unsafe_allow_html=True)

st.markdown('<div class="subtitle">KNN Machine Learning Model Deployment</div>',
            unsafe_allow_html=True)

# ----------------------------
# Input Section
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    transaction_amount = st.number_input(
        "💰 Transaction Amount", min_value=0.0)

    hour_of_day = st.slider(
        "⏰ Hour of Day", 0, 23)

    is_weekend = st.selectbox(
        "📅 Is Weekend?", [0, 1])

    num_items = st.number_input(
        "🛒 Number of Items", min_value=0)

    customer_age = st.number_input(
        "👤 Customer Age", min_value=1)

    prev_transactions = st.number_input(
        "📈 Previous Transactions", min_value=0)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    distance_from_home = st.number_input(
        "🏠 Distance From Home", min_value=0.0)

    device_type = st.number_input(
        "📱 Device Type", min_value=0)

    network_quality = st.number_input(
        "📶 Network Quality", min_value=0)

    is_first_transaction = st.selectbox(
        "🆕 First Transaction?", [0, 1])

    store_type = st.number_input(
        "🏪 Store Type", min_value=0)

    velocity_score = st.number_input(
        "⚡ Velocity Score", min_value=0.0)

    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Transaction"):

    data = pd.DataFrame([[
        transaction_amount,
        hour_of_day,
        is_weekend,
        num_items,
        customer_age,
        prev_transactions,
        distance_from_home,
        device_type,
        network_quality,
        is_first_transaction,
        store_type,
        velocity_score
    ]], columns=[
        'transaction_amount',
        'hour_of_day',
        'is_weekend',
        'num_items',
        'customer_age',
        'prev_transactions',
        'distance_from_home',
        'device_type',
        'network_quality',
        'is_first_transaction',
        'store_type',
        'velocity_score'
    ])

    prediction = model.predict(data)[0]

    st.write("")

    if prediction == 1:
        st.markdown(
            '<div class="result-danger">🚨 Fraudulent Transaction Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-success">✅ Genuine Transaction</div>',
            unsafe_allow_html=True
        )

    st.write("")

    st.metric(
        label="Prediction Result",
        value=str(prediction)
    )
