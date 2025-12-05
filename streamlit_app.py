# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "https://amazon-smart.onrender.com/recommend"

st.title("🛒 Smart Product Recommendations")

user_id = st.text_input("Enter User ID", value="U00001")
k = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    try:
        resp = requests.get(BACKEND_URL, params={"user_id": user_id, "k": k}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        recs = data.get("recommendations", [])
        st.success(f"Top {len(recs)} recommendations for {user_id}")

        # Optionally show product details if you have products.csv in same folder
        try:
            df = pd.read_csv("data/products.csv")  # adjust path if needed
            rec_df = df[df["product_id"].isin(recs)]
            st.dataframe(rec_df)
        except FileNotFoundError:
            st.write("Recommended product IDs:", recs)

    except Exception as e:
        st.error(f"Error calling backend: {e}")
