# app.py
# Smart Suggestion App — single-file Streamlit app
# - Search (press Enter) + price filters (supports 10k, 10,000)
# - Product grid + product details + content-based recommendations
# - Loads sample data if ./data/prod_meta.csv missing
# - Shows debug info; uses uploaded debug file path below

UPLOADED_DEBUG_FILE = "/mnt/data/eaef5e07-5416-44ce-90af-b3484e5b8768.json"

import streamlit as st
import pandas as pd
import numpy as np
import os, json, re, math
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
    vocab = [t for t,_ in terms_sorted[:max_features]]
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

def build_text_index_simple(df, text_cols=None, max_features=6000, id_col='item_id'):
    if text_cols is None:
        text_cols = ["title","description"]
    docs = []
    df2 = df.reset_index(drop=True)
    for _, row in df2.iterrows():
        parts = [str(row.get(c,"")) for c in text_cols]
        docs.append(" ".join(parts))
    vocab, mat = build_tfidf_matrix(docs, max_features=max_features)
    index_map = {i: df2.loc[i, id_col] for i in range(len(df2))}
    return {"vocab": vocab, "mat": mat, "index_map": index_map}

# -------------------------
# Load data
# -------------------------
@st.cache_resource
def load_prod_meta(path=os.path.join(BASE_DIR, "prod_meta.csv")):
    if not os.path.exists(path):
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
        # ensure price numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(int)
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
    # ensure price numeric even if CSV has commas or strings
    try:
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(',',''), errors='coerce').fillna(0).astype(int)
    except Exception:
        df['price'] = 0
    return df

prod_df = load_prod_meta()

# build TF-IDF index
@st.cache_resource
def build_text_index(df):
    return build_text_index_simple(df, text_cols=["title","description"], max_features=4000)

text_index = build_text_index(prod_df)
TFIDF_MAT = text_index['mat']
IDX_TO_ITEM = text_index['index_map']
ITEM_TO_IDX = {str(v): k for k,v in IDX_TO_ITEM.items()}

# -------------------------
# Helpers: parse human price text like "10k", "10,000"
# -------------------------
def parse_price_input_raw(s):
    if s is None:
        return 0
    stx = str(s).strip().lower()
    if stx == "" or stx in ["0","0.0"]:
        return 0
    stx = stx.replace(",", "").replace("₹","")
    # support "10k" and "10.5k"
    if stx.endswith("k"):
        try:
            return int(float(stx[:-1]) * 1000)
        except:
            return 0
    # support plain numbers
    try:
        return int(float(stx))
    except:
        return 0

# -------------------------
# Search and recommendation functions
# -------------------------
def search_products(query: str, min_price=None, max_price=None):
    q = str(query).strip().lower()
    df = prod_df.copy()
    if q:
        mask = df['title'].fillna('').str.lower().str.contains(q) | df['description'].fillna('').str.lower().str.contains(q)
        df = df[mask]
    if min_price is not None:
        try:
            df = df[df['price'] >= int(min_price)]
        except:
            pass
    if max_price is not None and max_price > 0:
        try:
            df = df[df['price'] <= int(max_price)]
        except:
            pass
    return df.reset_index(drop=True)

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
# UI Components (product card) - fixed nesting
# -------------------------
def product_card(parent_col, row):
    """
    Render into the provided parent_col to avoid deep nested columns.
    """
    parent_col.image(row["image_url"] if row["image_url"] else "https://via.placeholder.com/200x150.png?text=No+Image", width=180)
    parent_col.markdown(f"**{row['title']}**")
    parent_col.markdown(f"Brand: {row.get('brand','')}")
    try:
        price_int = int(row.get('price', 0))
    except Exception:
        price_int = 0
    parent_col.markdown(f"Price: ₹{price_int}")

    # create two buttons side-by-side inside this parent column (one-level nesting)
    bcol1, bcol2 = parent_col.columns([1,1])
    with bcol1:
        if bcol1.button("View", key=f"view_{row['item_id']}"):
            st.session_state['last_viewed'] = row["item_id"]
            st.session_state['recent_views'] = [row["item_id"]] + st.session_state.get('recent_views', [])
            st.session_state['recent_views'] = st.session_state['recent_views'][:30]
            st.experimental_rerun()
    with bcol2:
        if bcol2.button("Add to cart", key=f"add_{row['item_id']}"):
            st.session_state.setdefault('cart', {})
            st.session_state['cart'][row["item_id"]] = st.session_state['cart'].get(row["item_id"], 0) + 1
            st.success("Added to cart")

# -------------------------
# Session state defaults
# -------------------------
if "cart" not in st.session_state:
    st.session_state['cart'] = {}
if "recent_views" not in st.session_state:
    st.session_state['recent_views'] = []
if "last_viewed" not in st.session_state:
    st.session_state['last_viewed'] = None

# -------------------------
# Page layout + controls
# -------------------------
st.title("🧠 Smart Suggestion App")
st.markdown("Type a product name, set min/max price (supports `10k`, `10,000`) and press **Enter** or click **Search**.")

left, right = st.columns([1,3])

# Left column: optional assistant + debug
with left:
    st.header("Controls & Debug")
    st.markdown("Use the search box on the right for quick searching. Assistant removed to make search primary.")
    st.markdown("---")
    st.markdown("Dataset info")
    st.write(f"Rows: {prod_df.shape[0]}, Columns: {prod_df.shape[1]}")
    st.markdown("### Debug: sample rows")
    st.write(prod_df[['item_id','title','brand','price']].head(8))
    st.markdown("---")
    st.markdown("Uploaded debug file path")
    st.code(UPLOADED_DEBUG_FILE)
    if os.path.exists(UPLOADED_DEBUG_FILE):
        if st.button("Preview uploaded JSON"):
            try:
                with open(UPLOADED_DEBUG_FILE, "r") as f:
                    st.json(json.load(f))
            except Exception as e:
                st.write("Could not open debug file:", e)
    st.markdown("---")
    st.markdown("Cart")
    cart = st.session_state.get('cart', {})
    if cart:
        total = 0
        for iid, qty in cart.items():
            r = prod_df[prod_df['item_id']==iid]
            if not r.empty:
                p = int(r.iloc[0].get('price',0))
                st.write(f"{r.iloc[0]['title']} — {qty} x ₹{p} = ₹{qty*p}")
                total += qty*p
        st.markdown(f"**Total: ₹{total}**")
        if st.button("Checkout"):
            st.success("Checkout simulated — order placed (demo).")
            st.session_state['cart'] = {}
    else:
        st.write("Cart empty")

# Right column: search and results
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
    # parse price inputs
    min_price = parse_price_input_raw(min_raw)
    max_price = parse_price_input_raw(max_raw)
    min_price_bound = None if min_price == 0 else min_price
    max_price_bound = None if max_price == 0 else max_price

    # debug: show parsed values
    st.markdown(f"**Parsed price filter:** min = {min_price_bound}, max = {max_price_bound}")

    if submitted:
        results = search_products(q_text, min_price_bound, max_price_bound)
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

    # product details + recommendations
    lv = st.session_state.get('last_viewed', None)
    if lv:
        st.markdown("---")
        st.subheader("Product details")
        prow = prod_df[prod_df['item_id']==lv].iloc[0]
        st.image(prow['image_url'] if prow['image_url'] else 'https://via.placeholder.com/400x300.png?text=No+Image')
        st.markdown(f"### {prow['title']}")
        st.write(prow.get('description',''))
        st.write(f"Brand: {prow.get('brand','')}  — Price: ₹{int(prow.get('price',0))}")
        st.markdown("**Recommendations for this product**")
        recs = rec_content_sim(lv, top_k=6)
        if not recs:
            st.write("No content-based recommendations available.")
        else:
            for rid in recs:
                rrow = prod_df[prod_df['item_id']==rid].iloc[0]
                st.write(f"- {rrow['title']} — ₹{int(rrow.get('price',0))}")

# End
