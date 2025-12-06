# app.py — FINAL ENHANCED VERSION
import os
import re
import math
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
from pandas.errors import ParserError
import csv

st.set_page_config(page_title="Smart Suggestion App", layout="wide")

BASE_DIR = "./data"
CSV_PATH = os.path.join(BASE_DIR, "prod_meta.csv")
CLEANED_CSV_PATH = os.path.join(BASE_DIR, "prod_meta.cleaned.csv")
EXPECTED_FIELDS = ["item_id","title","category_id","brand","price","item_code","image_url","description"]

# ---------------- LOAD CSV ----------------
def repair_csv(in_path, out_path, expected=8):
    fixed = 0
    with open(in_path, newline='', encoding='utf-8') as fin, open(out_path, 'w', newline='', encoding='utf-8') as fout:
        r = csv.reader(fin); w = csv.writer(fout)
        for row in r:
            if len(row) == expected:
                w.writerow(row)
            elif len(row) < expected:
                w.writerow(row + [""] * (expected - len(row))); fixed += 1
            else:
                w.writerow(row[:expected-1] + [",".join(row[expected-1:])]); fixed += 1
    return fixed

@st.cache_resource
def load_df():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(columns=EXPECTED_FIELDS)
    try:
        df = pd.read_csv(CSV_PATH)
    except ParserError:
        repair_csv(CSV_PATH, CLEANED_CSV_PATH)
        df = pd.read_csv(CLEANED_CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    for c in EXPECTED_FIELDS:
        if c not in df.columns:
            df[c] = ""
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype(int)
    df["year"] = (
        df["description"]
        .str.extract(r"(20[2-9][0-9])")
        .fillna("2022")
        .astype(int)
        .clip(2020, 2025)
    )
    return df.reset_index(drop=True)

# ---------------- TF-IDF ----------------
def tokenize(t): return [w for w in re.findall(r"[a-z0-9]+", str(t).lower()) if len(w) > 1]

def build_tfidf(docs):
    doc_tok = [tokenize(x) for x in docs]
    freqs = []; dfreq = defaultdict(int)
    for t in doc_tok:
        c = Counter(t); freqs.append(c)
        for k in c: dfreq[k] += 1
    vocab = sorted(dfreq, key=dfreq.get, reverse=True)[:5000]
    idx = {t:i for i,t in enumerate(vocab)}
    n, m = len(docs), len(vocab)
    mat = np.zeros((n, m), float)
    idf = np.zeros(m, float)
    for t,i in idx.items(): idf[i] = math.log((1+n)/(1+dfreq[t])) + 1
    for r, c in enumerate(freqs):
        s = 0
        for t,f in c.items():
            if t in idx:
                val = f * idf[idx[t]]
                mat[r, idx[t]] = val; s += val*val
        if s > 0: mat[r] /= math.sqrt(s)
    return vocab, mat

def cosine_sim(mat, i): return mat.dot(mat[i])

@st.cache_resource
def build_text_index(df):
    docs = (df["title"] + " " + df["description"]).tolist()
    _, M = build_tfidf(docs)
    return {"mat": M, "index": {i: df.loc[i,"item_id"] for i in range(len(df))}}

# ---------------- CO-VIEW ----------------
@st.cache_resource
def load_coview():
    path = os.path.join(BASE_DIR, "co_view_top.json")
    if os.path.exists(path):
        try: return json.load(open(path, "r", encoding="utf-8"))
        except: return {}
    return {}

# ---------------- HELPERS ----------------
def parse_price(s):
    if not s: return 0
    s = s.lower().replace("₹","").replace(",","")
    if s.endswith("k"):
        try: return int(float(s[:-1]) * 1000)
        except: return 0
    try: return int(float(s))
    except: return 0

def search(df, q, cat, year, minp, maxp):
    q = q.lower().strip()
    out = df
    if q:
        mask = df["title"].str.lower().str.contains(q) | df["brand"].str.lower().str.contains(q) | df["category_id"].str.lower().str.contains(q)
        out = df[mask]

    if cat != "All":
        out = out[out["category_id"] == cat]
    if year != "All":
        out = out[out["year"] == int(year)]
    if minp: out = out[out["price"] >= minp]
    if maxp: out = out[out["price"] <= maxp]
    return out

def rec_similar(pid, index):
    i_map = index["index"]
    inv = {v:k for k,v in i_map.items()}
    if pid not in inv: return []
    sims = cosine_sim(index["mat"], inv[pid])
    order = sims.argsort()[::-1]
    out = []
    for o in order:
        if o == inv[pid]: continue
        out.append(i_map[o])
        if len(out) >= 6: break
    return out

def card(col, r):
    with col:
        st.image(r["image_url"] or "https://via.placeholder.com/200", width=180)
        st.markdown(f"**{r['title']}**")
        st.write(f"{r['brand']}  |  ₹{r['price']}")
        if st.button("👁 View", key=f"v_{r['item_id']}"):
            st.session_state["view"] = r["item_id"]; st.experimental_rerun()
        if st.button("🛒 Add", key=f"a_{r['item_id']}"):
            st.session_state["cart"][r["item_id"]] = st.session_state["cart"].get(r["item_id"],0)+1
            st.success("Added ✓")

# ---------------- SESSION ----------------
if "view" not in st.session_state: st.session_state["view"] = None
if "cart" not in st.session_state: st.session_state["cart"] = {}
if "recent" not in st.session_state: st.session_state["recent"] = []

# ---------------- LOAD ----------------
df = load_df()
text = build_text_index(df)
coview = load_coview()

# ---------------- UI ----------------
st.title("🧠 Smart Product Recommendation Engine")

with st.sidebar:
    st.header("Filters")
    q = st.text_input("Search")
    cat = st.selectbox("Category", ["All"] + sorted(df["category_id"].unique().tolist()))
    year = st.selectbox("Year", ["All"] + sorted(df["year"].unique().tolist()))
    minp = parse_price(st.text_input("Min price", "0"))
    maxp = parse_price(st.text_input("Max price", "0"))
    sort = st.selectbox("Sort by", ["Relevance", "Price high → low", "Price low → high", "Newest first"])

res = search(df, q, cat, year, minp, maxp)

if sort == "Price high → low": res = res.sort_values("price", ascending=False)
elif sort == "Price low → high": res = res.sort_values("price", ascending=True)
elif sort == "Newest first": res = res.sort_values("year", ascending=False)

st.subheader(f"{len(res)} products found")
cols = st.columns(3)
for i, (_, r) in enumerate(res.head(12).iterrows()):
    card(cols[i % 3], r)

# ---------------- DETAILS ----------------
if st.session_state["view"]:
    pid = st.session_state["view"]
    pr = df[df["item_id"] == pid].iloc[0]
    st.markdown("---")
    st.header(pr["title"])
    st.image(pr["image_url"], width=350)
    st.write(pr["description"])
    st.write(f"Brand: {pr['brand']} | ₹{pr['price']} | Year: {pr['year']}")

    # Save recent view
    if pid not in st.session_state["recent"]:
        st.session_state["recent"].insert(0, pid)
        st.session_state["recent"] = st.session_state["recent"][:10]

    st.subheader("People also buy:")
    for rid in coview.get(pid, [])[:5]:
        r = df[df["item_id"] == rid]
        if not r.empty:
            rr = r.iloc[0]
            st.write(f"- {rr['title']} — ₹{rr['price']}")

    st.subheader("Similar products:")
    for rid in rec_similar(pid, text):
        r = df[df["item_id"] == rid]
        if not r.empty:
            rr = r.iloc[0]
            st.write(f"- {rr['title']} — ₹{rr['price']}")

# ---------------- TRENDING ----------------
st.markdown("---")
st.subheader("🔥 Trending")
popular = df.sample(6, random_state=1)
cols = st.columns(3)
for i,(_,r) in enumerate(popular.iterrows()):
    card(cols[i % 3], r)

# ---------------- RECENTLY VIEWED ----------------
if st.session_state["recent"]:
    st.markdown("---")
    st.subheader("⏱ Recently viewed")
    cols = st.columns(3)
    for i, pid in enumerate(st.session_state["recent"][:6]):
        r = df[df["item_id"] == pid].iloc[0]
        card(cols[i % 3], r)

# ---------------- CART ----------------
st.sidebar.header("Cart")
total = 0
for pid, qty in st.session_state["cart"].items():
    r = df[df["item_id"] == pid]
    if not r.empty:
        pr = r.iloc[0]
        st.sidebar.write(f"{pr['title']} × {qty} = ₹{qty * pr['price']}")
        total += qty * pr['price']
st.sidebar.write(f"Total: ₹{total}")
