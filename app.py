# app.py
# Smart Suggestion App - corrected, robust version
# - Robust CSV loader that auto-cleans malformed rows (fixes ParserError)
# - Lightweight TF-IDF recommender (no sklearn)
# - Search form (Enter works), price parsing (10k etc.), product grid, product details, cart
# - Avoids deep Streamlit nesting issues

import os
import re
import math
import json
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
from pandas.errors import ParserError
import csv

# Local uploaded debug/log/image paths (from your environment)
UPLOADED_DEBUG_FILE = "/mnt/data/eaef5e07-5416-44ce-90af-b3484e5b8768.json"
MARKETING_IMAGE = "/mnt/data/WhatsApp Image 2025-11-04 at 20.27.51.jpeg"
LOG_PATH = "/mnt/data/logs-vaishcodes14-amazon-smart-main-app.py-2025-11-22T03_44_24.581Z.txt"

st.set_page_config(page_title="Smart Suggestion App", layout="wide")

BASE_DIR = "./data"
os.makedirs(BASE_DIR, exist_ok=True)
CSV_PATH = os.path.join(BASE_DIR, "prod_meta.csv")
CLEANED_CSV_PATH = os.path.join(BASE_DIR, "prod_meta.cleaned.csv")

# -------------------------
# TF-IDF helpers (no sklearn)
# -------------------------
def tokenize(text: Any) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in toks if len(t) > 1]

def build_tfidf_matrix(docs: List[str], max_features: int = 6000):
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
    vocab_index = {t: i for i, t in enumerate(vocab)}
    n_docs = len(docs)
    n_terms = len(vocab)
    if n_docs == 0 or n_terms == 0:
        return vocab, np.zeros((n_docs, n_terms), dtype=np.float32)
    idf = np.zeros(n_terms, dtype=np.float32)
    for t, i in vocab_index.items():
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

def cosine_sim_rows(mat: np.ndarray, query_idx: int) -> np.ndarray:
    if mat is None or mat.size == 0:
        return np.array([])
    q = mat[query_idx]
    sims = mat.dot(q)
    return sims

# -------------------------
# CSV cleaning helpers
# -------------------------
EXPECTED_FIELDS = ["item_id","title","category_id","brand","price","item_code","image_url","description"]

def repair_csv_join_extras(in_path: str, out_path: str, expected_cols_count: int = 8):
    """
    Reads CSV lines using csv.reader and if a row has more fields than expected,
    joins the extras into the last column (description).
    Writes cleaned CSV to out_path.
    """
    fixed = 0
    total = 0
    with open(in_path, newline='', encoding='utf-8') as fin, \
         open(out_path, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        for i, row in enumerate(reader, start=1):
            total += 1
            if len(row) == expected_cols_count:
                writer.writerow(row)
            elif len(row) < expected_cols_count:
                # pad missing fields
                row2 = row + [""]*(expected_cols_count - len(row))
                writer.writerow(row2[:expected_cols_count])
                fixed += 1
            else:
                # too many fields: join extras into the last column
                first = row[:expected_cols_count-1]
                last = ",".join(row[expected_cols_count-1:])
                first.append(last)
                writer.writerow(first)
                fixed += 1
    return {"total": total, "fixed": fixed, "out_path": out_path}

# -------------------------
# Load product metadata with robust fallback/repair
# -------------------------
@st.cache_resource
def load_prod_meta(path: str = CSV_PATH) -> pd.DataFrame:
    # If file missing -> create a small sample so the app shows something
    if not os.path.exists(path):
        sample = [
            ["p001","Pixel 6a","phones","Google",19999,1001,"https://via.placeholder.com/200x150.png?text=Pixel+6a","Google Pixel 6a smartphone."],
            ["p002","Galaxy A23","phones","Samsung",14999,1002,"https://via.placeholder.com/200x150.png?text=Galaxy+A23","Samsung A23 - 4GB RAM, 64GB."],
            ["p003","Mi Note 11","phones","Xiaomi",12999,1003,"https://via.placeholder.com/200x150.png?text=Mi+Note+11","Xiaomi Note 11 with large battery."],
            ["p004","OnePlus Nord CE","phones","OnePlus",17999,1004,"https://via.placeholder.com/200x150.png?text=OnePlus+Nord","OnePlus Nord CE suitable for gaming."],
            ["p005","Redmi 12C","phones","Xiaomi",8999,1005,"https://via.placeholder.com/200x150.png?text=Redmi+12C","Low budget phone with good battery."],
            ["p006","Realme Narzo","phones","Realme",10999,1006,"https://via.placeholder.com/200x150.png?text=Realme+Narzo","Realme Narzo series phone."]
        ]
        df = pd.DataFrame(sample, columns=EXPECTED_FIELDS)
        # normalize price
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
        return df

    # Try reading normally
    try:
        df = pd.read_csv(path)
    except ParserError as e:
        # try repairing CSV automatically and re-read
        repair_info = repair_csv_join_extras(path, CLEANED_CSV_PATH, expected_cols_count=len(EXPECTED_FIELDS))
        try:
            df = pd.read_csv(repair_info["out_path"])
            st.warning(f"prod_meta.csv had inconsistent rows. Auto-repaired {repair_info['fixed']} rows; using cleaned file.")
        except Exception as e2:
            # fallback to sample dataframe
            st.error(f"prod_meta.csv could not be parsed even after repair: {e2}. Falling back to sample data.")
            sample = [
                ["p001","Pixel 6a","phones","Google",19999,1001,"https://via.placeholder.com/200x150.png?text=Pixel+6a","Google Pixel 6a smartphone."]
            ]
            df = pd.DataFrame(sample, columns=EXPECTED_FIELDS)
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
            return df
    except Exception as e:
        st.error(f"Failed to read prod_meta.csv: {e}. Falling back to sample data.")
        sample = [
            ["p001","Pixel 6a","phones","Google",19999,1001,"https://via.placeholder.com/200x150.png?text=Pixel+6a","Google Pixel 6a smartphone."]
        ]
        df = pd.DataFrame(sample, columns=EXPECTED_FIELDS)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
        return df

    # Normalize columns and types
    df.columns = [c.strip() for c in df.columns]
    for c in EXPECTED_FIELDS:
        if c not in df.columns:
            df[c] = ""
    if "description" not in df.columns:
        df["description"] = df["title"].astype(str)
    # Clean price
    try:
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',',''), errors='coerce').fillna(0).astype(int)
    except Exception:
        df['price'] = 0
    return df

# -------------------------
# Build text index (cached)
# -------------------------
@st.cache_resource
def build_text_index(df: pd.DataFrame):
    docs = (df['title'].fillna('') + ' ' + df['description'].fillna('')).astype(str).tolist()
    vocab, mat = build_tfidf_matrix(docs, max_features=4000)
    index_map = {i: df.reset_index(drop=True).loc[i, 'item_id'] for i in range(len(df))}
    return {'vocab': vocab, 'mat': mat, 'index_map': index_map}

# -------------------------
# Price parsing helper (supports "10k", "10,000", "₹10k")
# -------------------------
def parse_price_input_raw(s: Any) -> int:
    if s is None:
        return 0
    stx = str(s).strip().lower()
    if stx == "" or stx in ["0","0.0"]:
        return 0
    stx = stx.replace(",", "").replace("₹", "").replace("inr", "")
    if stx.endswith("k"):
        try:
            return int(float(stx[:-1]) * 1000)
        except:
            return 0
    try:
        return int(float(stx))
    except:
        return 0

# -------------------------
# Search & recommend functions
# -------------------------
def search_products(df: pd.DataFrame, query: str, min_price: Optional[int] = None, max_price: Optional[int] = None) -> pd.DataFrame:
    q = str(query).strip().lower()
    out = df.copy()
    if q:
        mask = out['title'].fillna('').str.lower().str.contains(q) | out['description'].fillna('').str.lower().str.contains(q)
        out = out[mask]
    if min_price is not None:
        out = out[out['price'] >= int(min_price)]
    if max_price is not None and max_price > 0:
        out = out[out['price'] <= int(max_price)]
    return out.reset_index(drop=True)

def rec_content_sim(item_id: str, text_index: Dict[str, Any], top_k: int = 6) -> List[str]:
    idx_map = text_index['index_map']
    item_to_idx = {str(v): k for k, v in idx_map.items()}
    if str(item_id) not in item_to_idx:
        return []
    idx = item_to_idx[str(item_id)]
    mat = text_index['mat']
    sims = cosine_sim_rows(mat, idx)
    order = np.argsort(sims)[::-1]
    res = []
    for o in order:
        if o == idx:
            continue
        res.append(idx_map[o])
        if len(res) >= top_k:
            break
    return res

# -------------------------
# UI helpers - product card (avoid deep nesting)
# -------------------------
def product_card(parent_col, row):
    parent_col.image(row.get("image_url") if row.get("image_url") else "https://via.placeholder.com/200x150.png?text=No+Image", width=180)
    parent_col.markdown(f"**{row.get('title','')}**")
    parent_col.markdown(f"Brand: {row.get('brand','')}")
    try:
        price_int = int(row.get('price', 0))
    except Exception:
        price_int = 0
    parent_col.markdown(f"Price: ₹{price_int}")
    bcol1, bcol2 = parent_col.columns([1,1])
    with bcol1:
        if st.button("View", key=f"view_{row['item_id']}"):
            st.session_state['last_viewed'] = row['item_id']
            st.experimental_rerun()
    with bcol2:
        if st.button("Add to cart", key=f"add_{row['item_id']}"):
            st.session_state.setdefault('cart', {})
            st.session_state['cart'][row['item_id']] = st.session_state['cart'].get(row['item_id'], 0) + 1
            st.success("Added to cart")

# -------------------------
# Session defaults
# -------------------------
if "cart" not in st.session_state:
    st.session_state['cart'] = {}
if "last_viewed" not in st.session_state:
    st.session_state['last_viewed'] = None

# -------------------------
# Load data & index
# -------------------------
prod_df = load_prod_meta()
text_index = build_text_index(prod_df)
TFIDF_MAT = text_index['mat']
IDX_TO_ITEM = text_index['index_map']

# -------------------------
# Page layout
# -------------------------
st.title("🧠 Smart Suggestion App")
st.markdown("Type a product name, set min/max price (supports `10k`, `10,000`) and press **Enter** or click **Search**.")

left, right = st.columns([1,3])

with left:
    st.header("About / Marketing idea")
    # show uploaded marketing image if present
    if os.path.exists(MARKETING_IMAGE):
        try:
            st.image(MARKETING_IMAGE, caption="Marketing idea (uploaded image)")
        except Exception:
            st.info("Marketing image found but could not be displayed.")
    else:
        st.info("Marketing image not found at: " + MARKETING_IMAGE)
    st.markdown("**Idea:** Smart Product Recommendation Engine")
    st.markdown("**Problem:** Users can’t find what suits them best.")
    st.markdown("**Tech Stack:** Python, TensorFlow, AWS Personalize")
    st.markdown("**Workflow:** Train on purchase & view history → predict preferences → suggest top products.")
    st.markdown("**Relevance:** Boosts conversion rates with personalized shopping.")
    st.markdown("---")
    st.markdown("Dataset info")
    st.write(f"Rows: {prod_df.shape[0]}, Columns: {prod_df.shape[1]}")
    st.markdown("---")
    st.markdown("Uploaded debug file:")
    st.code(UPLOADED_DEBUG_FILE)
    if os.path.exists(LOG_PATH):
        st.markdown("Detected log file:")
        st.code(LOG_PATH)

with right:
    with st.form("search_form"):
        col_q, col_min, col_max, col_btn = st.columns([3,1,1,1])
        with col_q:
            q_text = st.text_input("Search products (e.g. 'mobile')", key="q_text")
        with col_min:
            min_raw = st.text_input("Min price (₹) — allow 10k", value="0", key="min_raw")
        with col_max:
            max_raw = st.text_input("Max price (₹) — allow 15k", value="0", key="max_raw")
        with col_btn:
            submitted = st.form_submit_button("Search")

    # parse prices
    min_price = parse_price_input_raw(min_raw)
    max_price = parse_price_input_raw(max_raw)
    min_price_bound = None if min_price == 0 else min_price
    max_price_bound = None if max_price == 0 else max_price

    st.markdown(f"**Parsed price filter:** min = {min_price_bound}, max = {max_price_bound}")

    if submitted:
        results = search_products(prod_df, q_text, min_price_bound, max_price_bound)
    else:
        results = prod_df.sort_values('price', ascending=False).reset_index(drop=True)

    st.subheader(f"Found {len(results)} products")
    if results.empty:
        st.write("No products found. Try a broader query or remove price filters.")
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(results.head(60).iterrows()):
            parent_col = cols[i % 3]
            product_card(parent_col, row)

    lv = st.session_state.get('last_viewed', None)
    if lv:
        st.markdown('---')
        st.subheader('Product details')
        prow = prod_df[prod_df['item_id'] == lv].iloc[0]
        st.image(prow.get('image_url') if prow.get('image_url') else 'https://via.placeholder.com/400x300.png?text=No+Image')
        st.markdown(f"### {prow.get('title')}")
        st.write(prow.get('description',''))
        st.write(f"Brand: {prow.get('brand','')} — Price: ₹{int(prow.get('price',0))}")
        st.markdown('**Recommendations for this product**')
        recs = rec_content_sim(lv, text_index, top_k=6)
        if not recs:
            st.write('No content-based recommendations available.')
        else:
            for rid in recs:
                rrow = prod_df[prod_df['item_id'] == rid].iloc[0]
                st.write(f"- {rrow.get('title')} — ₹{int(rrow.get('price',0))}")

# Sidebar debug & cart
st.sidebar.markdown('---')
st.sidebar.markdown('Debug: prod_df')
st.sidebar.write(f"rows: {prod_df.shape[0]}, cols: {prod_df.shape[1]}")
st.sidebar.write(prod_df.head(8))

st.sidebar.markdown('Cart')
cart = st.session_state.get('cart', {})
if cart:
    total = 0
    for iid, qty in cart.items():
        row = prod_df[prod_df['item_id'] == iid]
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

st.sidebar.markdown('Made with ❤️ — Smart Recommender Demo')
