"""Text similarity metrics using stdlib only (no numpy/sklearn).

Provides TF-IDF cosine similarity for measuring cross-turn repetitiveness.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization, stripping punctuation."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def term_frequencies(tokens: list[str]) -> dict[str, float]:
    """Normalized term frequency vector."""
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def idf(documents: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency across a corpus of tokenized documents."""
    n = len(documents)
    if n == 0:
        return {}
    doc_freq: Counter[str] = Counter()
    for doc in documents:
        doc_freq.update(set(doc))
    return {t: math.log((n + 1) / (df + 1)) + 1.0 for t, df in doc_freq.items()}


def tfidf_vector(tokens: list[str], idf_weights: dict[str, float]) -> dict[str, float]:
    """TF-IDF weighted vector for a single document."""
    tf = term_frequencies(tokens)
    return {t: freq * idf_weights.get(t, 1.0) for t, freq in tf.items()}


def cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts."""
    # Dot product over shared keys
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def pairwise_similarity(texts: list[str]) -> list[tuple[int, int, float]]:
    """Compute pairwise TF-IDF cosine similarity between a list of texts.

    Returns list of (i, j, similarity) for consecutive pairs and
    all-pairs where similarity > 0.
    """
    if len(texts) < 2:
        return []

    # Tokenize all
    tokenized = [tokenize(t) for t in texts]

    # Compute IDF across all documents
    idf_weights = idf(tokenized)

    # Build TF-IDF vectors
    vectors = [tfidf_vector(tok, idf_weights) for tok in tokenized]

    # Compute all consecutive pairs and any high-similarity pairs
    results: list[tuple[int, int, float]] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_sim(vectors[i], vectors[j])
            if sim > 0:
                results.append((i, j, round(sim, 4)))

    return results


def consecutive_similarity(texts: list[str]) -> list[float]:
    """Compute TF-IDF cosine similarity between each consecutive pair.

    Returns list of similarities: [sim(0,1), sim(1,2), ...]
    """
    if len(texts) < 2:
        return []

    tokenized = [tokenize(t) for t in texts]
    idf_weights = idf(tokenized)
    vectors = [tfidf_vector(tok, idf_weights) for tok in tokenized]

    return [
        round(cosine_sim(vectors[i], vectors[i + 1]), 4)
        for i in range(len(vectors) - 1)
    ]


def mean_similarity(texts: list[str]) -> float:
    """Mean pairwise TF-IDF cosine similarity across all text pairs."""
    pairs = pairwise_similarity(texts)
    if not pairs:
        return 0.0
    return round(sum(s for _, _, s in pairs) / len(pairs), 4)


def max_consecutive_similarity(texts: list[str]) -> float:
    """Maximum cosine similarity between any two consecutive texts."""
    sims = consecutive_similarity(texts)
    return max(sims) if sims else 0.0
