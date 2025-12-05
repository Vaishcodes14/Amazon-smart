import streamlit as st
import requests
import pandas as pd

# 🔗 Your backend on Render (FastAPI/Flask)
BACKEND_URL = "https://amazon-smart.onrender.com/recommend"

st.set_page_config(page_title="Smart Product Recommender", page_icon="🛒")
st.title("🛒 Amazon-Style Smart Product Recommendation Engine")
st.write("Enter a user ID to get personalized product recommendations from the backend API.")

# Inputs
user_id = st.text_input("User ID", value="U00001")
k = st.slider("Number of recommendations", 1, 20, 5)

if st.button("Get Recommendations"):
    if not user_id.strip():
        st.warning("Please enter a valid user ID.")
    else:
        with st.spinner("Contacting backend and fetching recommendations..."):
            try:
                # Call backend API
                params = {"user_id": user_id, "k": k}
                resp = requests.get(BACKEND_URL, params=params, timeout=15)

                if resp.status_code != 200:
                    st.error(f"Backend error: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    recs = data.get("recommendations", []) or data.get("recs", [])

                    if not recs:
                        st.warning("Backend returned no recommendations.")
                        st.json(data)
                    else:
                        st.success(f"Got {len(recs)} recommendations for user {user_id}")

                        # Try to show product details from products.csv (optional)
                        try:
                            products_df = pd.read_csv("data/products.csv")
                            rec_df = products_df[products_df["product_id"].isin(recs)]
                            if len(rec_df) > 0:
                                st.subheader("Recommended Products")
                                st.dataframe(rec_df)
                            else:
                                st.write("Recommended Product IDs:")
                                st.write(recs)
                        except Exception as e:
                            st.info("Could not load products.csv; showing only IDs.")
                            st.write("Recommended Product IDs:")
                            st.write(recs)

            except Exception as e:
                st.error(f"Failed to call backend: {e}")
