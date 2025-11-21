import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="Smart Product Recommender")

BASE_DIR = "./data"

# ---------------------------
# Load product metadata
# ---------------------------
@st.cache_resource
def load_prod_meta():
    path = os.path.join(BASE_DIR, "prod_meta.csv")
    if not os.path.exists(path):
        st.error("prod_meta.csv not found in /data folder. Upload it to run the app.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["item_id"] = df["item_id"].astype(str)

    # Ensure required columns exist
    for col in ["title", "brand", "category_id", "price", "image_url", "description"]:
        if col not in df.columns:
            df[col] = ""

    return df

prod_df = load_prod_meta()

# ---------------------------
# Load co-view and popular items
# ---------------------------
@st.cache_resource
def load_artifacts():
    arts = {}

    # co-view json
    cov_path = os.path.join(BASE_DIR, "co_view_top.json")
    if os.path.exists(cov_path):
        with open(cov_path, "r") as f:
            arts["co_view"] = json.load(f)
    else:
        arts["co_view"] = {}

    # popular items
    pop_path = os.path.join(BASE_DIR, "popular_items.joblib")
    if os.path.exists(pop_path):
        arts["popular"] = joblib.load(pop_path)
    else:
        arts["popular"] = []

    return arts

arts = load_artifacts()

# ---------------------------
# Build TF-IDF index
# ---------------------------
@st.cache_resource
def build_tfidf(df):
    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    vect = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    return vect, X

tfidf_vect, tfidf_matrix = build_tfidf(prod_df)

# Mapping item_id → index
itemid_to_index = {iid: idx for idx, iid in enumerate(prod_df["item_id"].tolist())}
index_to_itemid = {v: k for k, v in itemid_to_index.items()}

# ---------------------------
# Assistant filters
# ---------------------------
def parse_message(msg):
    msg = msg.lower()
    filters = {"budget_min": None, "budget_max": None, "brands": [], "categories": []}

    # budget
    import re
    m = re.search(r"under\s+([0-9,]+)", msg)
    if m:
        filters["budget_max"] = int(m.group(1).replace(",", ""))

    m = re.search(r"above\s+([0-9,]+)", msg)
    if m:
        filters["budget_min"] = int(m.group(1).replace(",", ""))

    # between
    m = re.search(r"between\s+([0-9,]+)\s+and\s+([0-9,]+)", msg)
    if m:
        filters["budget_min"] = int(m.group(1).replace(",", ""))
        filters["budget_max"] = int(m.group(2).replace(",", ""))

    # brand
    brands = prod_df["brand"].dropna().unique().tolist()
    for b in brands:
        if b.lower() in msg:
            filters["brands"].append(b)

    # category
    cats = prod_df["category_id"].dropna().unique().tolist()
    for c in cats:
        if c.lower() in msg:
            filters["categories"].append(c)

    return filters

# ---------------------------
# Recommendation engines
# ---------------------------

def rec_content_sim(item_id, top_k=20):
    """TF-IDF content similarity."""
    if item_id not in itemid_to_index:
        return []
    idx = itemid_to_index[item_id]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    order = sims.argsort()[::-1]
    results = []
    for o in order:
        if o == idx:
            continue
        results.append(index_to_itemid[o])
        if len(results) >= top_k:
            break
    return results

def rec_co_view(item_id, top_k=20):
    """Co-view suggestions based on JSON."""
    try:
        row = prod_df[prod_df["item_id"] == item_id].iloc[0]
        code = str(int(row["item_code"]))
        codes = arts["co_view"].get(code, [])[:top_k]
        ids = []
        for c in codes:
            sub = prod_df[prod_df["item_code"] == int(c)]
            if not sub.empty:
                ids.append(sub.iloc[0]["item_id"])
        return ids
    except:
        return []

def rec_popular(top_k=50):
    """Popular products fallback."""
    out = []
    for code in arts["popular"][:top_k]:
        sub = prod_df[prod_df["item_code"] == int(code)]
        if not sub.empty:
            out.append(sub.iloc[0]["item_id"])
    if not out:
        out = prod_df.sample(top_k)["item_id"].tolist()
    return out

# ---------------------------
# Merge final recommendations
# ---------------------------
def get_final_recommendations(query_filters, last_viewed, N=12):

    candidates = set()

    # Search related (content)
    if last_viewed:
        candidates.update(rec_content_sim(last_viewed, top_k=50))
        candidates.update(rec_co_view(last_viewed, top_k=50))

    # Popular as fallback
    candidates.update(rec_popular(100))

    df = prod_df[prod_df["item_id"].isin(candidates)].copy()
    if df.empty:
        return prod_df.sample(N)["item_id"].tolist()

    # apply filters
    if query_filters["budget_min"]:
        df = df[df["price"] >= query_filters["budget_min"]]

    if query_filters["budget_max"]:
        df = df[df["price"] <= query_filters["budget_max"]]

    if query_filters["brands"]:
        df = df[df["brand"].isin(query_filters["brands"])]

    if query_filters["categories"]:
        df = df[df["category_id"].isin(query_filters["categories"])]

    if df.empty:
        return prod_df.sample(N)["item_id"].tolist()

    return df.sample(min(N, len(df)))["item_id"].tolist()

# ---------------------------
# UI — Layout
# ---------------------------
st.title("🛒 Smart Product Recommendation Engine (Streamlit Cloud Ready)")

if "history" not in st.session_state:
    st.session_state.history = []

if "filters" not in st.session_state:
    st.session_state.filters = {"budget_min": None, "budget_max": None, "brands": [], "categories": []}

if "last_viewed" not in st.session_state:
    st.session_state.last_viewed = None

# ---------------------------
# Assistant panel
# ---------------------------
st.sidebar.header("💬 Shopping Assistant")

msg = st.sidebar.text_input("Ask anything… (e.g. gaming laptop under 100000)")

if st.sidebar.button("Apply"):
    st.session_state.history.append(("You", msg))
    parsed = parse_message(msg)
    st.session_state.filters.update(parsed)
    st.session_state.history.append(("Assistant", "Filters updated. Showing best results."))

for role, text in st.session_state.history[-6:]:
    st.sidebar.write(f"**{role}:** {text}")

if st.sidebar.button("Clear All Filters"):
    st.session_state.filters = {"budget_min": None, "budget_max": None, "brands": [], "categories": []}
    st.session_state.history.append(("Assistant", "All filters cleared."))

# ---------------------------
# Product listing
# ---------------------------
st.subheader("Recommended for You")

item_ids = get_final_recommendations(
    st.session_state.filters,
    st.session_state.last_viewed,
    N=12
)

cols = st.columns(3)

for i, iid in enumerate(item_ids):
    row = prod_df[prod_df["item_id"] == iid].iloc[0]
    with cols[i % 3]:
        st.image(row["image_url"], width=200)
        st.markdown(f"**{row['title']}**")
        st.write(f"{row['brand']} • ₹{int(row['price'])}")

        if st.button(f"View {iid}", key=f"view_{iid}"):
            st.session_state.last_viewed = iid
            st.experimental_rerun()

# ---------------------------
# Product detail
# ---------------------------
if st.session_state.last_viewed:
    st.markdown("---")
    st.subheader("🔍 Product Details")

    row = prod_df[prod_df["item_id"] == st.session_state.last_viewed].iloc[0]

    st.image(row["image_url"], width=400)
    st.markdown(f"### {row['title']}")
    st.write(f"**Brand:** {row['brand']}")
    st.write(f"**Category:** {row['category_id']}")
    st.write(f"**Price:** ₹{int(row['price'])}")
    st.write(row["description"])

    st.markdown("### Similar Products")
    sim_ids = rec_content_sim(st.session_state.last_viewed, top_k=6)
    for sid in sim_ids:
        sr = prod_df[prod_df["item_id"] == sid].iloc[0]
        st.write(f"- {sr['title']} — ₹{int(sr['price'])}")
