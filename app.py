# app.py
# Smart Suggestion App - corrected, robust version
# - Robust CSV loader that auto-cleans malformed rows (fixes ParserError)
# - Lightweight TF-IDF recommender (no sklearn)
# - Search form, price parsing, product grid, product details, cart
# - FIXED: no illegal nested column calls in Streamlit

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

UPLOADED_DEBUG_FILE = "/mnt/data/eaef5e07-5416-44ce-90af-b3484e5b8768.json"
MARKETING_IMAGE = "/mnt/data/WhatsApp Image 2025-11-04 at 20.27.51.jpeg"
LOG_PATH = "/mnt/data/logs-vaishcodes14-amazon-smart-main-app.py-2025-11-22T03_44_24.581Z.txt"

st.set_page_config(page_title="Smart Suggestion App", layout="wide")

BASE_DIR = "./data"
os.makedirs(BASE_DIR, exist_ok=True)
CSV_PATH = os.path.join(BASE_DIR, "prod_meta.csv")
CLEANED_CSV_PATH = os.path.join(BASE_DIR, "prod_meta.cleaned.csv")

# ---------------- TF-IDF HELPERS ----------------
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

# ---------------- CSV CLEANING ----------------
EXPECTED_FIELDS = ["item_id","title","category_id","brand","price","item_code","image_url","description"]

def repair_csv_join_extras(in_path: str, out_path: str, expected_cols_count: int = 8):
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
                row2 = row + [""]*(expected_cols_count - len(row))
                writer.writerow(row2[:expected_cols_count])
                fixed += 1
            else:
                first = row[:expected_cols_count-1]
                last = ",".join(row[expected_cols_count-1:])
                first.append(last)
                writer.writerow(first)
                fixed += 1
    return {"total": total, "fixed": fixed, "out_path": out_path}

# ---------------- LOAD & CLEAN CSV ----------------
@st.cache_resource
def load_prod_meta(path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        sample = [
            ["p001","Pixel 6a","phones","Google",19999,1001,"https://via.placeholder.com/200x150.png?text=Pixel+6a","Google Pixel 6a smartphone."],
            ["p002","Galaxy A23","phones","Samsung",14999,1002,"https://via.placeholder.com/200x150.png?text=Galaxy+A23","Samsung A23 - 4GB RAM, 64GB."],
        ]
        df = pd.DataFrame(sample, columns=EXPECTED_FIELDS)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
        return df
    try:
        df = pd.read_csv(path)
    except ParserError:
        repair_info = repair_csv_join_extras(path, CLEANED_CSV_PATH, expected_cols_count=len(EXPECTED_FIELDS))
        try:
            df = pd.read_csv(repair_info["out_path"])
            st.warning(f"prod_meta.csv repaired {repair_info['fixed']} rows.")
        except:
            df = pd.DataFrame([], columns=EXPECTED_FIELDS)
            return df
    df.columns = [c.strip() for c in df.columns]
    for c in EXPECTED_FIELDS:
        if c not in df.columns:
            df[c] = ""
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',',''), errors='coerce').fillna(0).astype(int)
    return df

# ---------------- BUILD TEXT INDEX ----------------
@st.cache_resource
def build_text_index(df: pd.DataFrame):
    docs = (df['title'].fillna('') + ' ' + df['description'].fillna('')).astype(str).tolist()
    vocab, mat = build_tfidf_matrix(docs, max_features=4000)
    index_map = {i: df.reset_index(drop=True).loc[i, 'item_id'] for i in range(len(df))}
    return {'vocab': vocab, 'mat': mat, 'index_map': index_map}

# ---------------- HELPERS ----------------
def parse_price_input_raw(s: Any) -> int:
    if s is None:
        return 0
    stx = str(s).strip().lower()
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

def search_products(df, query, min_price=None, max_price=None):
    q = str(query).strip().lower()
    out = df.copy()
    if q:
        mask = out['title'].fillna('').str.lower().str.contains(q) | out['description'].fillna('').str.lower().str.contains(q)
        out = out[mask]
    if min_price is not None:
        out = out[out['price'] >= min_price]
    if max_price is not None and max_price > 0:
        out = out[out['price'] <= max_price]
    return out.reset_index(drop=True)

def rec_content_sim(item_id: str, text_index, top_k=6):
    idx_map = text_index['index_map']
    item_to_idx = {str(v): k for k, v in idx_map.items()}
    if item_id not in item_to_idx:
        return []
    idx = item_to_idx[item_id]
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

# ---------------- FIXED PRODUCT CARD ----------------
def product_card(parent_col, row):
    parent_col.image(row.get("image_url") or "https://via.placeholder.com/200x150.png?text=No+Image", width=180)
    parent_col.markdown(f"**{row.get('title','')}**")
    parent_col.markdown(f"Brand: {row.get('brand','')}")
    price = int(row.get('price', 0))
    parent_col.markdown(f"Price: ₹{price}")

    # FIXED: no nested columns
    with parent_col:
        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            if st.button("View", key=f"view_{row['item_id']}"):
                st.session_state['last_viewed'] = row['item_id']
                st.experimental_rerun()
        with bcol2:
            if st.button("Add to cart", key=f"add_{row['item_id']}"):
                st.session_state.setdefault('cart', {})
                st.session_state['cart'][row['item_id']] = st.session_state['cart'].get(row['item_id'], 0) + 1
                st.success("Added to cart")

# ---------------- SESSION ----------------
if "cart" not in st.session_state:
    st.session_state['cart'] = {}
if "last_viewed" not in st.session_state:
    st.session_state['last_viewed'] = None

prod_df = load_prod_meta()
text_index = build_text_index(prod_df)

# ---------------- UI ----------------
st.title("🧠 Smart Suggestion App")
st.markdown("Search product & get similar recommendations!")

left, right = st.columns([1,3])

with left:
    st.header("Marketing Idea")
    if os.path.exists(MARKETING_IMAGE):
        st.image(MARKETING_IMAGE, caption="Marketing Idea")
    st.write(f"Dataset size: {prod_df.shape[0]} rows")

with right:
    with st.form("search_form"):
        col_q, col_min, col_max, col_btn = st.columns([3,1,1,1])
        q_text = col_q.text_input("Search product")
        min_raw = col_min.text_input("Min ₹ (supports 10k)", "0")
        max_raw = col_max.text_input("Max ₹ (supports 10k)", "0")
        submitted = col_btn.form_submit_button("Search")

    min_p = parse_price_input_raw(min_raw)
    max_p = parse_price_input_raw(max_raw)
    min_b = None if min_p == 0 else min_p
    max_b = None if max_p == 0 else max_p

    results = search_products(prod_df, q_text, min_b, max_b) if submitted else prod_df
    st.subheader(f"{len(results)} products found")

    cols = st.columns(3)
    for i, (_, row) in enumerate(results.head(60).iterrows()):
        product_card(cols[i % 3], row)

    lv = st.session_state.get('last_viewed')
    if lv:
        st.markdown("---")
        prow = prod_df[prod_df['item_id'] == lv].iloc[0]
        st.image(prow.get("image_url") or "https://via.placeholder.com/400")
        st.subheader(prow.get("title"))
        st.write(prow.get("description"))
        st.write(f"Brand: {prow.get('brand','')} — ₹{int(prow.get('price',0))}")
        st.markdown("**Similar products**")
        recs = rec_content_sim(lv, text_index, top_k=6)
        for rid in recs:
            rrow = prod_df[prod_df['item_id'] == rid].iloc[0]
            st.write(f"- {rrow['title']} — ₹{int(rrow['price'])}")

# ---------------- SIDEBAR CART ----------------
st.sidebar.header("Cart")
cart = st.session_state['cart']
if cart:
    total = 0
    for iid, qty in cart.items():
        row = prod_df[prod_df['item_id'] == iid]
        if not row.empty:
            price = int(row.iloc[0]['price'])
            st.sidebar.write(f"{row.iloc[0]['title']} — {qty} × ₹{price} = ₹{qty*price}")
            total += qty * price
    st.sidebar.write(f"**Total: ₹{total}**")
    if st.sidebar.button("Checkout"):
        st.sidebar.success("Order placed (demo)")
        st.session_state['cart'] = {}
else:
    st.sidebar.write("Cart is empty")

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ Smart Recommender Demo")
