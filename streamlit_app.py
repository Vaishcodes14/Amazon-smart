import streamlit as st
import requests
import pandas as pd

# 🌐 YOUR BACKEND ON RENDER
BACKEND_URL = "https://amazon-smart.onrender.com"   # <- only this, nothing extra

st.set_page_config(page_title="Smart Product Recommender", page_icon="🛒")
st.title("🛒 Amazon Smart Product Recommendation Engine")
st.write("Type a user id and I will ask the backend for recommendations.")

user_id = st.text_input("User ID", value="U00001")
k = st.slider("Number of recommendations", 1, 20, 5)

if st.button("Get Recommendations"):
    if not user_id.strip():
        st.warning("Please enter a user id.")
    else:
        with st.spinner("Talking to backend..."):
            try:
                # try calling your backend
                resp = requests.get(
                    BACKEND_URL,
                    params={"user_id": user_id, "k": k},
                    timeout=15
                )
                if resp.status_code != 200:
                    st.error(f"Backend error: {resp.status_code} - {resp.text}")
                else:
                    data = resp.json()
                    # try common keys
                    recs = data.get("recommendations") or data.get("recs") or data.get("products") or []
                    if not recs:
                        st.write("Raw response from backend:")
                        st.json(data)
                    else:
                        st.success(f"Got {len(recs)} recommendations for {user_id}")

                        # show IDs + (optional) product details
                        st.write("Recommended product IDs:", recs)

                        try:
                            products_df = pd.read_csv("data/products.csv")
                            rec_df = products_df[products_df["product_id"].isin(recs)]
                            if len(rec_df) > 0:
                                st.subheader("Recommended Products Details")
                                st.dataframe(rec_df)
                        except Exception:
                            pass

            except Exception as e:
                st.error(f"Could not call backend: {e}")
