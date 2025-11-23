# smart_suggestion_app.py
# Simple, single-file Streamlit app — Smart Suggestion App
# - Primary action: Search + filters (press Enter or Search button to update)
# - Shows product grid and product details with content-based recommendations
# - Loads sample products if ./data/prod_meta.csv is missing
# - Displays uploaded debug file path from your environment

UPLOADED_DEBUG_FILE = "/mnt/data/eaef5e07-5416-44ce-90af-b3484e5b8768.json"

import streamlit as st
import pandas as pd
import numpy as np
import os, json
import re, math
from collections import Counter, defaultdict

st.set_page_config(page_title="Smart Suggestion App", layout="wide")

BASE_DIR = "./data"

# -------------------------
# Lightweight TF-IDF helpers
# -------------------------

def tokenize(text):
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in toks if len(t) > 1]


def build_tfidf_matrix(docs, max_features=6000):
    doc_tokens = [tokenize(d) for d in docs]
    term_doc_count = defaultdict(int)
    term_freqs = []
    for tokens in doc_tokens:
        c = Counter(tokens)
        term_freqs.append(c)
        for t in c.keys():
            term_doc_count[t] += 1
    terms_sorted = sorted(term_doc_count.items(), key=lambda x: x[1], reverse=True)
    vocab = [t for t, _ in terms_sorted[:max_features]]
    vocab_index = {t:i for i,t in enumerate(vocab)}
    n_docs = len(docs)
    n_terms = len(vocab)
    if n_docs == 0 or n_terms == 0:
        return vocab, np.zeros((n_docs, n_terms), dtype=np.float32)
    idf = np.zeros(n_terms, dtype=np.float32)
    for t,i in vocab_index.items():
        df = term_doc_count.get(t, 0)
        idf[i] = math.log((1 + n_docs) / (1 + df)) + 1.0
    mat = np.zeros((n_docs, n_terms), dtype=np.float32)
    for d_idx, c in enumerate(term_freqs):
        norm = 0.0
        for t, freq in c.items():
            if t in vocab_index:
                idx = vocab_index[t]
                tfidf = freq * idf[idx]
                mat[d_idx, idx] = tfidf
                norm += tfidf * tfidf
        if norm > 0:
            mat[d_idx, :] = mat[d_idx, :] / math.sqrt(norm)
    return vocab, mat


def cosine_sim_rows(mat, query_idx):
    if mat is None or mat.size == 0:
        return np.array([])
    q = mat[query_idx]
    sims = mat.dot(q)
    return sims

# -------------------------
# Data loading
# -------------------------

@st.cache_resource
def load_prod_meta(path=os.path.join(BASE_DIR, "prod_meta.csv")):
    if not os.path.exists(path):
        # sample dataset so the app immediately demonstrates behavior
        cols = ["item_id","title","category_id","brand","price","item_code","image_url","description"]
        sample = [
            ["p001","Pixel 6a","phones","Google",19999,1001,"https://via.placeholder.com/200x150.png?text=Pixel+6a","Google Pixel 6a smartphone."],
            ["p002","Galaxy A23","phones","Samsung",14999,1002,"https://via.placeholder.com/200x150.png?text=Galaxy+A23","Samsung A23 - 4GB RAM, 64GB."],
            ["p003","Mi Note 11","phones","Xiaomi",12999,1003,"https://via.placeholder.com/200x150.png?text=Mi+Note+11","Xiaomi Note 11 with large battery."],
            ["p004","OnePlus Nord CE","phones","OnePlus",17999,1004,"https://via.placeholder.com/200x150.png?text=OnePlus+Nord","OnePlus Nord CE suitable for gaming."],
            ["p005","Redmi 12C","phones","Xiaomi",8999,1005,"https://via.placeholder.com/200x150.png?text=Redmi+12C","Low budget phone with good battery."],
            ["p006","Realme Narzo","phones","Realme",10999,1006,"https://via.placeholder.com/200x150.png?text=Realme+Narzo","Realme Narzo series phone."]
        ]
        df = pd.DataFrame(sample, columns=cols)
        return df
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in ["item_id","title","category_id","brand","price","item_code"]:
        if c not in df.columns:
            df[c] = ""
    if "image_url" not in df.columns:
        df["image_url"] = ""
    if "description" not in df.columns:
        df["description"] = df["title"]
    try:
        df["item_code"] = df["item_code"].astype(int)
    except Exception:
        pass
    # ensure price numeric
    try:
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
    except Exception:
        df['price'] = 0
    return df

prod_df = load_prod_meta()

# build content index
@st.cache_resource
def build_text_index(df):
    docs = (df['title'].fillna('') + ' ' + df['description'].fillna('')).astype(str).tolist()
    vocab, mat = build_tfidf_matrix(docs, max_features=4000)
    index_map = {i: df.reset_index(drop=True).loc[i, 'item_id'] for i in range(len(df))}
    return {'vocab': vocab, 'mat': mat, 'index_map': index_map}

text_index = build_text_index(prod_df)
TFIDF_MAT = text_index['mat']
IDX_TO_ITEM = text_index['index_map']
ITEM_TO_IDX = {str(v): k for k,v in IDX_TO_ITEM.items()}

# -------------------------
# Simple search + filter function
# -------------------------

def search_products(query: str, min_price=None, max_price=None):
    q = str(query).strip().lower()
    df = prod_df.copy()
    if q:
        mask = df['title'].fillna('').str.lower().str.contains(q) | df['description'].fillna('').str.lower().str.contains(q)
        df = df[mask]
    if min_price is not None:
        df = df[df['price'] >= int(min_price)]
    if max_price is not None and max_price>0:
        df = df[df['price'] <= int(max_price)]
    return df.reset_index(drop=True)

# -------------------------
# Content-based recommendations
# -------------------------

def rec_content_sim(item_id, top_k=6):
    sid = str(item_id)
    if sid not in ITEM_TO_IDX:
        return []
    idx = ITEM_TO_IDX[sid]
    sims = cosine_sim_rows(TFIDF_MAT, idx)
    order = np.argsort(sims)[::-1]
    res = []
    for o in order:
        if o == idx:
            continue
        res.append(IDX_TO_ITEM[o])
        if len(res) >= top_k:
            break
    return res

# -------------------------
# UI
# -------------------------

# Top header
st.title("🧠 Smart Suggestion App")
st.markdown("Search for products using the box below — set min/max price and press **Enter** or click **Search**.")

# Layout: left sidebar for debug/info, right main for search/results
sidebar, main = st.columns([1,4])

with sidebar:
    st.header("Controls")
    st.markdown("**Dataset**")
    st.write(f"Rows: {prod_df.shape[0]}, Columns: {prod_df.shape[1]}")
    st.markdown("---")
    st.markdown("Uploaded debug file:")
    st.code(UPLOADED_DEBUG_FILE)
    if os.path.exists(UPLOADED_DEBUG_FILE):
        if st.button("Preview debug JSON"):
            try:
                with open(UPLOADED_DEBUG_FILE,'r') as f:
                    st.json(json.load(f))
            except Exception as e:
                st.write("Could not open debug file:", e)
    st.markdown("---")
    st.markdown("Tip: Type product name, set price filters, press Enter or Search.")

with main:
    with st.form("search_form"):
        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            query = st.text_input("Search products (e.g. 'mobile')", key='q')
        with col2:
            min_price = st.number_input("Min price (₹)", min_value=0, value=0, step=500)
        with col3:
            max_price = st.number_input("Max price (₹)", min_value=0, value=0, step=500)
        submitted = st.form_submit_button("Search")
    # if user presses Enter or clicks Search, submitted becomes True
    if submitted:
        results = search_products(query, None if min_price==0 else min_price, None if max_price==0 else max_price)
    else:
        # initial view: show popular / all sorted by price desc (demo)
        results = prod_df.sort_values('price', ascending=False).reset_index(drop=True)

    st.subheader(f"Found {len(results)} products")
    if results.empty:
        st.write("No products found. Try a broader query or remove price filters.")
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(results.head(30).iterrows()):
            with cols[i % 3]:
                st.image(row['image_url'] if row['image_url'] else 'https://via.placeholder.com/200x150.png?text=No+Image', width=180)
                st.markdown(f"**{row['title']}**")
                st.write(row.get('brand',''))
                st.write(f"₹{int(row.get('price',0))}")
                if st.button("View", key=f"view_{row['item_id']}"):
                    st.session_state['last_viewed'] = row['item_id']
                    st.experimental_rerun()
                if st.button("Add to cart", key=f"add_{row['item_id']}"):
                    st.session_state.setdefault('cart', {})
                    st.session_state['cart'][row['item_id']] = st.session_state['cart'].get(row['item_id'],0) + 1
                    st.success('Added to cart')

    # product details + recommendations
    lv = st.session_state.get('last_viewed', None)
    if lv:
        st.markdown('---')
        st.subheader('Product details')
        prow = prod_df[prod_df['item_id']==lv].iloc[0]
        st.image(prow['image_url'] if prow['image_url'] else 'https://via.placeholder.com/400x300.png?text=No+Image')
        st.markdown(f"### {prow['title']}")
        st.write(prow.get('description',''))
        st.write(f"Brand: {prow.get('brand','')} — Price: ₹{int(prow.get('price',0))}")
        st.markdown('**Recommendations for this product**')
        recs = rec_content_sim(lv, top_k=6)
        if not recs:
            st.write('No content-based recommendations available.')
        else:
            for rid in recs:
                rrow = prod_df[prod_df['item_id']==rid].iloc[0]
                st.write(f"- {rrow['title']} — ₹{int(rrow.get('price',0))}")

# Footer: cart info
st.sidebar.markdown('---')
cart = st.session_state.get('cart', {})
if cart:
    st.sidebar.markdown('### Cart')
    total = 0
    for iid, qty in cart.items():
        row = prod_df[prod_df['item_id']==iid]
        if not row.empty:
            p = int(row.iloc[0].get('price',0))
            st.sidebar.write(f"{row.iloc[0]['title']} — {qty} x ₹{p} = ₹{qty*p}")
            total += qty*p
    st.sidebar.markdown(f"**Total: ₹{total}**")
    if st.sidebar.button('Checkout'):
        st.sidebar.success('Checkout simulated — order placed (demo).')
        st.session_state['cart'] = {}
else:
    st.sidebar.write('Cart empty')

# End of file
