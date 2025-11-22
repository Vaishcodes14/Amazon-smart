# app.py
# Hybrid Amazon-Full UI with ChatGPT-style Assistant (left) + Product Grid (right)
# Expects artifacts folder content under ./data/
#
# Required files (place in amazon/data/):
# - artifacts/prod_meta.csv               (or data/prod_meta.csv)
# - artifacts/als_model.joblib            (optional; if missing, ALS fallback used)
# - artifacts/user_le.joblib
# - artifacts/item_le.joblib
# - artifacts/user_item_matrix.joblib
# - artifacts/item_user_matrix.joblib
# - artifacts/co_view_top.json
# - artifacts/popular_items.joblib
#
# NOTE: If you trained in Colab and downloaded artifacts.zip, unzip and put the artifact files into ./data/
#
# Debug / uploaded file path (from your environment). This is the path you uploaded earlier.
UPLOADED_DEBUG_FILE = "/mnt/data/eaef5e07-5416-44ce-90af-b3484e5b8768.json"

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, time, random
from typing import List

st.set_page_config(layout="wide", page_title="Amazon-Full — Smart Recommender", initial_sidebar_state="collapsed")

# Where artifacts will live in your repo
BASE_DIR = "./data"  # change if your data folder is different

# -------------------------
# Lightweight TF-IDF + Cosine Similarity (NO sklearn)
# -------------------------
import re, math
from collections import Counter, defaultdict

def tokenize(text):
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in toks if len(t) > 1]

def build_tfidf_matrix(docs, max_features=6000):
    """
    docs: list[str]
    returns: vocab (list[str]), mat (numpy float32 n_docs x n_terms) with row normalization
    """
    doc_tokens = [tokenize(d) for d in docs]
    term_doc_count = defaultdict(int)  # number of docs containing term
    term_freqs = []
    for tokens in doc_tokens:
        c = Counter(tokens)
        term_freqs.append(c)
        for t in c.keys():
            term_doc_count[t] += 1

    # choose top terms by document frequency
    terms_sorted = sorted(term_doc_count.items(), key=lambda x: x[1], reverse=True)
    vocab = [t for t, _ in terms_sorted[:max_features]]
    vocab_index = {t: i for i, t in enumerate(vocab)}

    n_docs = len(docs)
    n_terms = len(vocab)

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

def cosine_sim_rows(mat, query_idx):
    """
    mat: n_docs x n_terms (rows normalized)
    query_idx: row index
    returns: similarity vector (n_docs,)
    """
    q = mat[query_idx]
    sims = mat.dot(q)
    return sims

def build_text_index_simple(df, text_cols=None, max_features=6000, id_col='item_id'):
    if text_cols is None:
        text_cols = ["title", "description"]
    docs = []
    df2 = df.reset_index(drop=True)
    for _, row in df2.iterrows():
        parts = [str(row.get(c, "")) for c in text_cols]
        docs.append(" ".join(parts))
    vocab, mat = build_tfidf_matrix(docs, max_features=max_features)
    index_map = {i: df2.loc[i, id_col] for i in range(len(df2))}
    return {"vocab": vocab, "mat": mat, "index_map": index_map}

# -------------------------
# Utilities & loading
# -------------------------
@st.cache_resource
def load_prod_meta(path=os.path.join(BASE_DIR, "prod_meta.csv")):
    if not os.path.exists(path):
        st.warning(f"prod_meta.csv not found at {path}. Upload artifacts/prod_meta.csv to use full features.")
        # return empty df with expected columns
        cols = ["item_id","title","category_id","brand","price","item_code","image_url","description"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # ensure required columns exist
    for c in ["item_id","title","category_id","brand","price","item_code"]:
        if c not in df.columns:
            df[c] = ""
    if "image_url" not in df.columns:
        df["image_url"] = ""
    if "description" not in df.columns:
        df["description"] = df["title"]
    # coerce types where possible
    try:
        df["item_code"] = df["item_code"].astype(int)
    except Exception:
        pass
    return df

@st.cache_resource
def load_artifacts(base_dir=BASE_DIR):
    art = {}
    # Try load optional artifacts. Missing artifacts are fine — app falls back.
    def safe_load(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    # joblib artifacts
    art["als"] = safe_load(os.path.join(base_dir, "als_model.joblib"))
    art["user_le"] = safe_load(os.path.join(base_dir, "user_le.joblib"))
    art["item_le"] = safe_load(os.path.join(base_dir, "item_le.joblib"))
    art["user_item_matrix"] = safe_load(os.path.join(base_dir, "user_item_matrix.joblib"))
    art["item_user_matrix"] = safe_load(os.path.join(base_dir, "item_user_matrix.joblib"))
    art["popular"] = safe_load(os.path.join(base_dir, "popular_items.joblib")) or []
    # co-view json
    try:
        with open(os.path.join(base_dir, "co_view_top.json"), "r") as f:
            art["co_view"] = json.load(f)
    except Exception:
        art["co_view"] = {}
    return art

# load data
prod_df = load_prod_meta()
arts = load_artifacts()

# -------------------------
# build lightweight TF-IDF index for content similarity (title + description)
# -------------------------
@st.cache_resource
def build_text_index(df):
    # returns dict with 'mat' and 'index_map'
    return build_text_index_simple(df, text_cols=["title", "description"], max_features=6000)

text_index = build_text_index(prod_df)
tfidf_matrix = text_index["mat"] if text_index is not None else None
# mapping: index -> item_id, and reverse map
index_to_itemid = text_index["index_map"] if text_index is not None else {}
itemid_to_idx = {v: k for k, v in index_to_itemid.items()}
idx_to_itemid = {k: v for k, v in index_to_itemid.items()}

# Session state defaults
if "cart" not in st.session_state:
    st.session_state.cart = {}
if "recent_views" not in st.session_state:
    st.session_state.recent_views = []
if "assistant_history" not in st.session_state:
    st.session_state.assistant_history = []
if "filters" not in st.session_state:
    st.session_state.filters = {"budget_min": None, "budget_max": None, "brands": [], "use_case": None, "categories": []}
if "last_viewed" not in st.session_state:
    st.session_state.last_viewed = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# -------------------------
# Assistant parsing & scoring
# -------------------------
def parse_assistant_message(text: str):
    text_l = text.lower()
    filters = {}
    import re
    m = re.search(r"under\s+([0-9,]+)", text_l) or re.search(r"below\s+([0-9,]+)", text_l)
    if m:
        filters["budget_max"] = int(m.group(1).replace(",", ""))
    m = re.search(r"above\s+([0-9,]+)", text_l)
    if m:
        filters["budget_min"] = int(m.group(1).replace(",", ""))
    m = re.search(r"between\s+([0-9,]+)\s+and\s+([0-9,]+)", text_l)
    if m:
        filters["budget_min"] = int(m.group(1).replace(",", ""))
        filters["budget_max"] = int(m.group(2).replace(",", ""))
    use_cases = ["gaming", "office", "photography", "video", "travel", "student", "streaming", "work", "editing"]
    for uc in use_cases:
        if uc in text_l:
            filters["use_case"] = uc
            break
    brands = prod_df["brand"].dropna().unique().tolist()
    mentioned = [b for b in brands if b.lower() in text_l]
    if mentioned:
        filters["brands"] = mentioned[:6]
    cats = prod_df["category_id"].dropna().unique().tolist()
    cat_mentioned = [c for c in cats if c.lower() in text_l]
    if cat_mentioned:
        filters["categories"] = cat_mentioned[:6]
    return filters

def assistant_score(df: pd.DataFrame, f: dict):
    scores = np.ones(len(df), dtype=float)
    if f.get("budget_min") is not None:
        scores *= np.where(df["price"] >= f["budget_min"], 1.0, 0.3)
    if f.get("budget_max") is not None:
        scores *= np.where(df["price"] <= f["budget_max"], 1.0, 0.3)
    if f.get("brands"):
        bset = set([b.lower() for b in f["brands"]])
        scores *= np.where(df["brand"].str.lower().isin(bset), 1.3, 1.0)
    if f.get("categories"):
        cset = set([c.lower() for c in f["categories"]])
        scores *= np.where(df["category_id"].str.lower().isin(cset), 1.2, 1.0)
    if f.get("use_case"):
        uc = f["use_case"].lower()
        keywords = {
            "gaming": ["gaming","rtx","geforce","ryzen","144hz","240hz","gpu"],
            "photography": ["camera","lens","dslr","mirrorless","megapixel","stabiliz"],
            "video": ["4k","video","stabiliz","gimbal","microphone"],
            "travel": ["lightweight","compact","portable","battery"],
            "student": ["budget","student","lightweight","ssd","office"]
        }
        kw = keywords.get(uc, [])
        if kw:
            txt = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
            match_score = np.zeros(len(df))
            for k in kw:
                match_score += txt.str.contains(k).astype(float)
            if match_score.max() > 0:
                match_score = 1 + (match_score / (match_score.max()))
                scores *= match_score
    return scores

# -------------------------
# Candidate retrieval helpers
# -------------------------
def recommend_with_als(user_id, top_k=25):
    try:
        if arts["als"] is None or arts["user_le"] is None or arts["item_le"] is None or arts["user_item_matrix"] is None:
            return []
        if user_id not in arts["user_le"].classes_:
            return []
        u_idx = int(arts["user_le"].transform([user_id])[0])
        recs = arts["als"].recommend(userid=u_idx, user_items=arts["user_item_matrix"][u_idx], N=top_k)
        codes = [int(r[0]) for r in recs]
        ids = [arts["item_le"].inverse_transform([c])[0] for c in codes]
        return ids
    except Exception:
        return []

def co_view_for_item(item_id, top_k=20):
    try:
        if not arts["co_view"]:
            return []
        row = prod_df[prod_df["item_id"]==item_id]
        if row.empty:
            return []
        code = str(int(row.iloc[0]["item_code"]))
        codes = arts["co_view"].get(code, [])[:top_k]
        if arts["item_le"] is not None:
            try:
                ids = [arts["item_le"].inverse_transform([int(c)])[0] for c in codes]
                return ids
            except Exception:
                pass
        ids = []
        for c in codes:
            rows = prod_df[prod_df["item_code"]==int(c)]
            if not rows.empty:
                ids.append(rows.iloc[0]["item_id"])
        return ids
    except Exception:
        return []

def content_similar(item_id, top_k=20):
    try:
        if tfidf_matrix is None or item_id not in itemid_to_idx:
            return []
        idx = itemid_to_idx[item_id]
        sims = cosine_sim_rows(tfidf_matrix, idx)
        order = np.argsort(sims)[::-1]
        results = []
        for o in order:
            if o == idx:
                continue
            results.append(idx_to_itemid[o])
            if len(results) >= top_k:
                break
        return results
    except Exception:
        return []

def popular_candidates(top_k=50):
    res = []
    for p in arts.get("popular", [])[:top_k]:
        try:
            if arts.get("item_le") is not None:
                iid = arts["item_le"].inverse_transform([int(p)])[0]
                res.append(iid)
        except Exception:
            rows = prod_df[prod_df["item_code"]==int(p)]
            if not rows.empty:
                res.append(rows.iloc[0]["item_id"])
    if not res:
        res = prod_df.sort_values("price", ascending=False)["item_id"].head(top_k).tolist()
    return res

# -------------------------
# Merge candidates & rerank
# -------------------------
def merge_and_score(user_id: str, current_item: str, assistant_filters: dict, N=12):
    candidates = []
    # ALS
    als_ids = recommend_with_als(user_id, top_k=50)
    for i,iid in enumerate(als_ids):
        candidates.append((iid, {"als_score": float(1.0/(1+i+1)), "source":"als"}))
    # co-view
    if current_item:
        cov = co_view_for_item(current_item, top_k=50)
        for i,iid in enumerate(cov):
            candidates.append((iid, {"cov_score": float(1.0/(1+i+1)), "source":"co"}))
    # content sim
    if current_item:
        sim = content_similar(current_item, top_k=50)
        for i,iid in enumerate(sim):
            candidates.append((iid, {"sim_score": float(1.0/(1+i+1)), "source":"sim"}))
    # popular
    pop = popular_candidates(200)
    for i,iid in enumerate(pop):
        candidates.append((iid, {"pop_score": float(1.0/(1+i+1)), "source":"pop"}))

    # aggregate
    agg = {}
    for iid, sc in candidates:
        if iid not in agg:
            agg[iid] = sc
        else:
            for k,v in sc.items():
                agg[iid][k] = agg[iid].get(k,0) + v

    ids = list(agg.keys())
    df_cands = prod_df[prod_df["item_id"].isin(ids)].reset_index(drop=True)
    if df_cands.empty:
        return popular_candidates(N)

    as_scores = assistant_score(df_cands, assistant_filters)
    as_map = dict(zip(df_cands["item_id"].tolist(), as_scores.tolist()))

    cand_list = []
    for iid, info in agg.items():
        als_s = info.get("als_score", 0)
        sim_s = info.get("sim_score", 0)
        cov_s = info.get("cov_score", 0)
        pop_s = info.get("pop_score", 0)
        assistant_s = as_map.get(iid, 1.0)
        final = 0.30*als_s + 0.25*sim_s + 0.20*cov_s + 0.15*pop_s + 0.30*(assistant_s-1)
        cand_list.append((iid, final))

    cand_list.sort(key=lambda x: x[1], reverse=True)

    # hard filters
    filtered = []
    for iid, score in cand_list:
        row = prod_df[prod_df["item_id"]==iid].iloc[0]
        if assistant_filters.get("budget_min") is not None and row["price"] < assistant_filters["budget_min"]:
            continue
        if assistant_filters.get("budget_max") is not None and row["price"] > assistant_filters["budget_max"]:
            continue
        if assistant_filters.get("brands"):
            if row["brand"].lower() not in [b.lower() for b in assistant_filters["brands"]]:
                continue
        if assistant_filters.get("categories"):
            if row["category_id"].lower() not in [c.lower() for c in assistant_filters["categories"]]:
                continue
        filtered.append((iid, score))
        if len(filtered) >= N:
            break

    if len(filtered) < N:
        for p in pop:
            if p not in [x[0] for x in filtered]:
                filtered.append((p, 0.0))
            if len(filtered) >= N:
                break
    return [x[0] for x in filtered[:N]]

# -------------------------
# UI components
# -------------------------
def product_card(row):
    st.image(row["image_url"] if row["image_url"] else "https://via.placeholder.com/200x150.png?text=No+Image", width=180)
    st.markdown(f"**{row['title']}**")
    st.markdown(f"Brand: {row.get('brand','')}")
    try:
        price_int = int(row.get('price', 0))
    except Exception:
        price_int = 0
    st.markdown(f"Price: ₹{price_int}")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("View", key=f"view_{row['item_id']}"):
            st.session_state.last_viewed = row["item_id"]
            st.session_state.recent_views.insert(0, row["item_id"])
            if len(st.session_state.recent_views) > 30:
                st.session_state.recent_views = st.session_state.recent_views[:30]
            st.experimental_rerun()
    with col2:
        if st.button("Add to cart", key=f"add_{row['item_id']}"):
            st.session_state.cart[row["item_id"]] = st.session_state.cart.get(row["item_id"], 0) + 1
            st.success("Added to cart")

# -------------------------
# Layout
# -------------------------
left, right = st.columns([1,3])

with left:
    st.header("🧠 Shopping Assistant")
    st.markdown("Ask me what you're looking for, e.g.: *'Best gaming laptop under 100000 for streaming'*")
    for role, text in st.session_state.assistant_history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")

    user_msg = st.text_input("Ask the assistant...", key="assistant_input")
    col_a, col_b = st.columns([3,1])
    with col_b:
        if st.button("Send"):
            if user_msg and user_msg.strip():
                st.session_state.assistant_history.append(("user", user_msg))
                new_filters = parse_assistant_message(user_msg)
                st.session_state.filters.update(new_filters)
                reply_lines = []
                if "budget_max" in new_filters or "budget_min" in new_filters:
                    bm = new_filters.get("budget_min")
                    bM = new_filters.get("budget_max")
                    if bm and bM:
                        reply_lines.append(f"Got it — budget set between ₹{bm} and ₹{bM}.")
                    elif bM:
                        reply_lines.append(f"Okay, I'll look for items under ₹{bM}.")
                    elif bm:
                        reply_lines.append(f"Okay, I'll look for items above ₹{bm}.")
                if "use_case" in new_filters:
                    reply_lines.append(f"Use-case: **{new_filters['use_case']}** — I'll favor items suited for that.")
                if "brands" in new_filters and new_filters["brands"]:
                    reply_lines.append("I will prioritize these brands: " + ", ".join(new_filters["brands"]))
                if "categories" in new_filters and new_filters["categories"]:
                    reply_lines.append("Focusing on categories: " + ", ".join(new_filters["categories"]))
                if not reply_lines:
                    reply_lines = ["Okay — searching based on your message."]
                assistant_reply = " ".join(reply_lines)
                st.session_state.assistant_history.append(("assistant", assistant_reply))
                st.session_state.assistant_input = ""
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("Quick Filters")
    vmin, vmax = st.columns(2)
    with vmin:
        vmin_val = st.number_input("Min price (₹)", value=st.session_state.filters.get("budget_min") or 0, min_value=0)
    with vmax:
        vmax_val = st.number_input("Max price (₹)", value=st.session_state.filters.get("budget_max") or 0, min_value=0)
    st.session_state.filters["budget_min"] = vmin_val if vmin_val>0 else None
    st.session_state.filters["budget_max"] = vmax_val if vmax_val>0 else None

    cats = sorted(prod_df["category_id"].dropna().unique().tolist())
    chosen_cats = st.multiselect("Categories", options=cats, default=st.session_state.filters.get("categories") or [])
    st.session_state.filters["categories"] = chosen_cats

    brands = sorted(prod_df["brand"].dropna().unique().tolist())
    chosen_brands = st.multiselect("Brands", options=brands[:200], default=st.session_state.filters.get("brands") or [])
    st.session_state.filters["brands"] = chosen_brands

    if st.button("Clear assistant filters"):
        st.session_state.filters = {"budget_min": None, "budget_max": None, "brands": [], "use_case": None, "categories": []}
        st.session_state.assistant_history.append(("assistant", "Cleared filters. How else can I help?"))
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Cart")
    if st.session_state.cart:
        total = 0
        for iid, qty in st.session_state.cart.items():
            row = prod_df[prod_df["item_id"]==iid]
            if not row.empty:
                try:
                    price = int(row.iloc[0]["price"])
                except Exception:
                    price = 0
                st.write(f"{row.iloc[0]['title']} — {qty} x ₹{price} = ₹{qty*price}")
                total += qty*price
        st.markdown(f"**Total: ₹{total}**")
        if st.button("Checkout"):
            st.success("Checkout simulated — order placed (demo).")
            st.session_state.cart = {}
    else:
        st.write("Cart is empty")

with right:
    st.header("Products")
    q = st.text_input("Search products (keywords, e.g. 'gaming laptop i7')", key="search_q")
    sort_opt = st.selectbox("Sort by", options=["Relevance (assistant)","Price: Low to High","Price: High to Low","Newest"], index=0)
    per_page = st.selectbox("Show", options=[12, 24, 48], index=0)

    current_item = st.session_state.get("last_viewed", None)
    assistant_filters = st.session_state.filters.copy()
    if q and len(q.strip())>0:
        parsed = parse_assistant_message(q)
        assistant_filters.update(parsed)
    user_id = st.session_state.get("user_id", None)

    candidates = merge_and_score(user_id, current_item, assistant_filters, N=per_page)
    if q and q.strip():
        ql = q.lower()
        cand_rows = prod_df[prod_df["item_id"].isin(candidates)].copy()
        cand_rows["match"] = (cand_rows["title"].str.lower().str.contains(ql) | cand_rows["description"].str.lower().str.contains(ql)).astype(int)
        cand_rows = cand_rows.sort_values(["match"], ascending=False)
    else:
        cand_rows = prod_df[prod_df["item_id"].isin(candidates)].copy()

    if sort_opt == "Price: Low to High":
        cand_rows = cand_rows.sort_values("price", ascending=True)
    elif sort_opt == "Price: High to Low":
        cand_rows = cand_rows.sort_values("price", ascending=False)

    cols = st.columns(3)
    for i, (_, row) in enumerate(cand_rows.head(per_page).iterrows()):
        with cols[i % 3]:
            product_card(row)

    if current_item:
        st.markdown("---")
        st.subheader("Product details")
        row = prod_df[prod_df["item_id"]==current_item].iloc[0]
        left_col, mid_col, right_col = st.columns([2,3,2])
        with left_col:
            st.image(row["image_url"] if row["image_url"] else "https://via.placeholder.com/400x300.png?text=No+Image")
        with mid_col:
            st.markdown(f"### {row['title']}")
            st.markdown(f"**Brand:** {row['brand']}  •  **Category:** {row['category_id']}")
            try:
                price_display = int(row['price'])
            except Exception:
                price_display = 0
            st.markdown(f"**Price:** ₹{price_display}")
            st.markdown(row.get("description",""))
            if st.button("Add to cart (product page)"):
                st.session_state.cart[current_item] = st.session_state.cart.get(current_item,0) + 1
                st.success("Added to cart")
        with right_col:
            st.markdown("**Recommendations for this product**")
            sim = content_similar(current_item, top_k=6)
            for s in sim:
                r = prod_df[prod_df["item_id"]==s].iloc[0]
                try:
                    rp = int(r['price'])
                except Exception:
                    rp = 0
                st.write(f"- {r['title']} — ₹{rp}")

# Footer / debug
st.sidebar.markdown("### App status")
st.sidebar.write("Hybrid Amazon-Full demo — Assistant + ALS + content-sim + co-view merge")
st.sidebar.write("Artifacts detected:")
for k,v in arts.items():
    st.sidebar.write(f"- {k}: {'Loaded' if v else 'Missing'}")

# show uploaded debug file path and offer to preview
st.sidebar.markdown("---")
st.sidebar.markdown("Debug: uploaded file (raw) path")
st.sidebar.code(UPLOADED_DEBUG_FILE)
if st.sidebar.button("Preview uploaded file"):
    try:
        import json
        with open(UPLOADED_DEBUG_FILE, "r") as f:
            data = json.load(f)
        st.sidebar.write(data if isinstance(data, dict) else str(data)[:1000])
    except Exception as e:
        st.sidebar.write("Could not open uploaded debug file:", e)

st.sidebar.markdown("Made with ❤️ — Smart Recommender Demo")
