import streamlit as st
import pickle
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="KNN Prediction App",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #e8f5e9;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load Model
with open("knn_neighbour.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🤖 KNN Classification Prediction System")
st.markdown("### Enter the details below")

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
    hour_of_day = st.number_input("Hour of Day", min_value=0, max_value=23)
    is_weekend = st.selectbox("Is Weekend", [0, 1])
    num_items = st.number_input("Number of Items", min_value=0)
    customer_age = st.number_input("Customer Age", min_value=1)

with col2:
    prev_transactions = st.number_input("Previous Transactions", min_value=0)
    distance_from_home = st.number_input("Distance From Home", min_value=0.0)
    device_type = st.number_input("Device Type", min_value=0)
    network_quality = st.number_input("Network Quality", min_value=0)
    is_first_transaction = st.selectbox("First Transaction", [0, 1])
    store_type = st.number_input("Store Type", min_value=0)
    velocity_score = st.number_input("Velocity Score", min_value=0.0)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "num_items": num_items,
        "customer_age": customer_age,
        "prev_transactions": prev_transactions,
        "distance_from_home": distance_from_home,
        "device_type": device_type,
        "network_quality": network_quality,
        "is_first_transaction": is_first_transaction,
        "store_type": store_type,
        "velocity_score": velocity_score
    }])

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="prediction-box">
        Prediction: {prediction}
        </div>
        """,
        unsafe_allow_html=True
    )
