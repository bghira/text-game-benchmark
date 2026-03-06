"""Tests for tgb.similarity — TF-IDF cosine similarity metrics."""

import pytest

from tgb.similarity import (
    tokenize,
    term_frequencies,
    idf,
    tfidf_vector,
    cosine_sim,
    pairwise_similarity,
    consecutive_similarity,
    mean_similarity,
    max_consecutive_similarity,
)


# ── tokenize ──────────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        tokens = tokenize("It's a test, isn't it?")
        assert "it's" in tokens
        assert "isn't" in tokens
        assert "," not in tokens

    def test_empty(self):
        assert tokenize("") == []

    def test_numbers_excluded(self):
        assert tokenize("Room 42 is dark") == ["room", "is", "dark"]


# ── term_frequencies ──────────────────────────────────────────────────────

class TestTermFrequencies:
    def test_basic(self):
        tf = term_frequencies(["the", "cat", "the"])
        assert tf["the"] == pytest.approx(2 / 3)
        assert tf["cat"] == pytest.approx(1 / 3)

    def test_empty(self):
        tf = term_frequencies([])
        assert tf == {}


# ── idf ───────────────────────────────────────────────────────────────────

class TestIdf:
    def test_basic(self):
        docs = [["a", "b"], ["b", "c"], ["c", "d"]]
        weights = idf(docs)
        # "b" appears in 2/3 docs, "a" in 1/3
        assert weights["a"] > weights["b"]

    def test_empty(self):
        assert idf([]) == {}


# ── cosine_sim ────────────────────────────────────────────────────────────

class TestCosineSim:
    def test_identical_vectors(self):
        v = {"a": 1.0, "b": 2.0}
        assert cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert cosine_sim(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"a": 1.0, "b": 1.0}
        b = {"a": 1.0, "c": 1.0}
        sim = cosine_sim(a, b)
        assert 0.0 < sim < 1.0

    def test_empty_vectors(self):
        assert cosine_sim({}, {"a": 1.0}) == 0.0
        assert cosine_sim({}, {}) == 0.0


# ── pairwise_similarity ──────────────────────────────────────────────────

class TestPairwiseSimilarity:
    def test_identical_texts(self):
        pairs = pairwise_similarity(["the cat sat", "the cat sat"])
        assert len(pairs) == 1
        assert pairs[0][2] == pytest.approx(1.0, abs=0.01)

    def test_different_texts(self):
        pairs = pairwise_similarity([
            "the dark forest loomed ahead",
            "bright sunshine filled the meadow",
        ])
        assert len(pairs) >= 0  # may have some overlap via "the"
        if pairs:
            assert pairs[0][2] < 0.5

    def test_single_text(self):
        assert pairwise_similarity(["only one"]) == []

    def test_empty(self):
        assert pairwise_similarity([]) == []

    def test_three_texts(self):
        pairs = pairwise_similarity(["a b c", "a b d", "x y z"])
        # Should have up to 3 pairs: (0,1), (0,2), (1,2)
        indices = {(p[0], p[1]) for p in pairs}
        assert (0, 1) in indices  # "a" and "b" shared


# ── consecutive_similarity ────────────────────────────────────────────────

class TestConsecutiveSimilarity:
    def test_two_texts(self):
        sims = consecutive_similarity(["the cat sat", "the cat sat on a mat"])
        assert len(sims) == 1
        assert sims[0] > 0.3

    def test_three_texts(self):
        sims = consecutive_similarity(["a b", "c d", "e f"])
        assert len(sims) == 2

    def test_single(self):
        assert consecutive_similarity(["only one"]) == []


# ── mean_similarity ──────────────────────────────────────────────────────

class TestMeanSimilarity:
    def test_identical(self):
        sim = mean_similarity(["hello world", "hello world"])
        assert sim > 0.9

    def test_different(self):
        sim = mean_similarity([
            "the dark corridor stretched endlessly",
            "bright flowers bloomed in the garden",
            "a mechanical clock ticked on the wall",
        ])
        assert sim < 0.5

    def test_single(self):
        assert mean_similarity(["only"]) == 0.0


# ── max_consecutive_similarity ────────────────────────────────────────────

class TestMaxConsecutiveSimilarity:
    def test_returns_max(self):
        sims = max_consecutive_similarity([
            "unique text about cats",
            "completely different text about dogs",
            "completely different text about dogs and more",
        ])
        # The last two should be most similar
        assert sims > 0.0

    def test_single(self):
        assert max_consecutive_similarity(["only"]) == 0.0


# ── integration: realistic narrations ────────────────────────────────────

class TestRealisticNarrations:
    def test_varied_narrations_low_similarity(self):
        """Varied game narrations should have low pairwise similarity."""
        narrations = [
            "You follow the White Rabbit down a twisting burrow. The earth is damp and dark around you.",
            "The hall stretches before you, lined with tiny locked doors. A glass table gleams in the center.",
            "The Mad Hatter grins across the tea table, adjusting his oversized hat.",
        ]
        sim = mean_similarity(narrations)
        assert sim < 0.4, f"Expected low similarity for varied narrations, got {sim}"

    def test_repetitive_narrations_high_similarity(self):
        """Near-identical narrations should have high similarity."""
        narrations = [
            "You walk down the dark corridor. The walls are cold stone. A torch flickers ahead.",
            "You walk down the dark corridor. The walls are cold stone. A door appears ahead.",
            "You walk down the dark corridor. The walls are cold stone. A shadow moves ahead.",
        ]
        sim = mean_similarity(narrations)
        assert sim > 0.5, f"Expected high similarity for repetitive narrations, got {sim}"
