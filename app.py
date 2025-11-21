# ---- Begin: lightweight TF-IDF + cosine helpers (no scikit-learn) ----
import math
from collections import Counter, defaultdict

def tokenize(text):
    # simple tokenizer: lowercase, split on non-alpha, remove short tokens
    import re
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in toks if len(t) > 1]

def build_tfidf_matrix(docs, max_features=6000):
    """
    docs: list of string documents
    returns: vocab_list, tfidf_matrix (numpy float32 shape: n_docs x n_terms)
    """
    # token counts per doc
    doc_tokens = [tokenize(d) for d in docs]
    # global term counts
    term_doc_count = defaultdict(int)  # number of docs containing term
    term_freqs = []
    for tokens in doc_tokens:
        c = Counter(tokens)
        term_freqs.append(c)
        for t in c.keys():
            term_doc_count[t] += 1

    # keep top terms by document frequency (to limit features)
    terms_sorted = sorted(term_doc_count.items(), key=lambda x: x[1], reverse=True)
    vocab = [t for t,_ in terms_sorted[:max_features]]
    vocab_index = {t:i for i,t in enumerate(vocab)}
    n_docs = len(docs)
    n_terms = len(vocab)

    # compute idf
    idf = np.zeros(n_terms, dtype=np.float32)
    for t, i in vocab_index.items():
        df = term_doc_count.get(t, 0)
        idf[i] = math.log((1 + n_docs) / (1 + df)) + 1.0

    # build tf-idf matrix
    mat = np.zeros((n_docs, n_terms), dtype=np.float32)
    for d_idx, c in enumerate(term_freqs):
        # compute raw tf
        norm = 0.0
        for t, freq in c.items():
            if t in vocab_index:
                idx = vocab_index[t]
                tf = freq  # raw count
                mat[d_idx, idx] = tf * idf[idx]
                norm += (mat[d_idx, idx] ** 2)
        if norm > 0:
            mat[d_idx, :] = mat[d_idx, :] / (math.sqrt(norm))
    return vocab, mat

def cosine_sim_rows(mat, query_idx):
    """
    mat: n_docs x n_terms float32 (rows are normalized)
    query_idx: index of row to compute similarity against
    returns: numpy array of similarity scores
    """
    q = mat[query_idx]
    sims = mat.dot(q)  # since rows are normalized, dot product is cosine
    return sims

# wrapper to build the "index" once
def build_text_index_simple(df, text_cols=None, max_features=6000):
    if text_cols is None:
        text_cols = ["title", "description"]
    docs = []
    for _, row in df.iterrows():
        parts = []
        for c in text_cols:
            parts.append(str(row.get(c,"")))
        docs.append(" ".join(parts))
    vocab, mat = build_tfidf_matrix(docs, max_features=max_features)
    return {"vocab": vocab, "mat": mat, "index_map": {i: row["item_id"] for i,row in df.reset_index().iterrows()}}
# ---- End: lightweight TF-IDF + cosine helpers ----
