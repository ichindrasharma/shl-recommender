
"""
Improved Lightweight TF-IDF Retriever
Fast + better ranking quality
No torch
No sentence-transformers
"""

import json
import os
import re

import faiss
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------

_vectorizer = None
_index = None
_catalog = []


# ---------------------------------------------------------
# TECH KEYWORDS
# ---------------------------------------------------------

TECH_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "react",
    "node",
    "backend",
    "frontend",
    "developer",
    "software",
    "coding",
    "programming",
    "sql",
    "api",
    "cloud",
    "aws",
    "docker",
    "kubernetes",
    "devops",
    "engineering",
}


# ---------------------------------------------------------
# BUILD SEARCH TEXT
# ---------------------------------------------------------

def _build_text(item):

    parts = []

    # Strong boost for title
    if item.get("name"):
        parts.extend([item["name"]] * 4)

    # Medium boost for description
    if item.get("description"):
        parts.extend([item["description"]] * 2)

    # Medium boost for job levels
    if item.get("job_levels"):
        parts.extend(item["job_levels"])

    # Strong boost for keys/tags
    if item.get("keys"):
        parts.extend(item["keys"] * 3)

    return " ".join(parts)


# ---------------------------------------------------------
# LOAD CATALOG
# ---------------------------------------------------------

def load_catalog(path="catalog.json"):

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# INITIALIZE
# ---------------------------------------------------------

def initialize(path="catalog.json"):

    global _vectorizer
    global _index
    global _catalog

    print("Loading catalog...")

    _catalog = load_catalog(path)

    texts = [
        _build_text(item)
        for item in _catalog
    ]

    print("Building TF-IDF vectors...")

    _vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=7000,
        ngram_range=(1, 2),
    )

    matrix = _vectorizer.fit_transform(texts)

    vectors = matrix.toarray().astype("float32")

    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]

    _index = faiss.IndexFlatIP(dim)

    _index.add(vectors)

    print(f"Retriever ready with {len(_catalog)} items")


# ---------------------------------------------------------
# QUERY BOOSTING
# ---------------------------------------------------------

def _keyword_boost(query, item):

    score = 0.0

    query_words = set(
        re.findall(r"\w+", query.lower())
    )

    item_text = _build_text(item).lower()

    for word in query_words:

        # Boost exact keyword matches
        if word in item_text:
            score += 0.05

        # Extra boost for tech keywords
        if word in TECH_KEYWORDS and word in item_text:
            score += 0.15

    return score


# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------

def search(query, k=10):

    global _vectorizer
    global _index
    global _catalog

    query_vector = _vectorizer.transform([query])

    query_vector = (
        query_vector.toarray()
        .astype("float32")
    )

    faiss.normalize_L2(query_vector)

    scores, indices = _index.search(
        query_vector,
        min(k * 3, len(_catalog))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = dict(_catalog[idx])

        # Add keyword boost
        boosted_score = (
            float(score)
            + _keyword_boost(query, item)
        )

        item["score"] = boosted_score

        # Filter weak matches
        if boosted_score < 0.12:
            continue

        results.append(item)

    # Sort after boosting
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Deduplicate by URL/link
    seen = set()
    final_results = []

    for item in results:

        url = item.get("url") or item.get("link")

        if not url:
            continue

        if url in seen:
            continue

        seen.add(url)

        final_results.append(item)

        if len(final_results) >= k:
            break

    return final_results


# ---------------------------------------------------------
# GET BY NAME
# ---------------------------------------------------------

def get_by_name(name):

    name = name.lower()

    for item in _catalog:

        item_name = item.get(
            "name",
            ""
        ).lower()

        if name in item_name:
            return item

    return None


# ---------------------------------------------------------
# GET ALL
# ---------------------------------------------------------

def get_all():

    return _catalog

