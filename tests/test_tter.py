"""Unit tests for asr_correction.tter."""

import pytest


@pytest.fixture(scope="module")
def compute_tter():
    from asr_correction.tter import compute_tter
    return compute_tter


def test_perfect_hypothesis_has_zero_tter(compute_tter):
    ref = "Laura deployed the service on Cloudflare yesterday"
    hyp = "Laura deployed the service on Cloudflare yesterday"
    target_terms = {"Laura", "Cloudflare"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["tter"] == 0.0
    assert result["substitutions"] == 0
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["target_total"] == 2


def test_substitution_on_target_term_counts(compute_tter):
    ref = "Laura deployed to Cloudflare"
    hyp = "Laura deployed to Cloudware"  # Cloudflare → Cloudware
    target_terms = {"Laura", "Cloudflare"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["substitutions"] == 1
    assert result["target_total"] == 2
    assert result["tter"] == pytest.approx(0.5)


def test_substitution_on_non_target_word_is_ignored(compute_tter):
    ref = "Laura deployed the service"
    hyp = "Laura deployed a service"  # "the" → "a", not a target
    target_terms = {"Laura"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["substitutions"] == 0
    assert result["tter"] == 0.0


def test_deletion_of_target_term_counts(compute_tter):
    ref = "Laura and David met"
    hyp = "Laura met"  # David deleted
    target_terms = {"Laura", "David"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["deletions"] == 1
    assert result["target_total"] == 2
    assert result["tter"] == pytest.approx(0.5)


def test_insertion_of_target_shaped_word_counts(compute_tter):
    # "Kubernetes" inserted in hypothesis; it's a target term by name.
    ref = "we deployed the service"
    hyp = "we deployed Kubernetes the service"
    target_terms = {"Kubernetes"}
    result = compute_tter(ref, hyp, target_terms)
    # No target terms in ref → target_total = 0 → tter is defined as 0.0 by convention
    # but insertions are still tracked.
    assert result["insertions"] == 1
    assert result["target_total"] == 0
    assert result["tter"] == 0.0  # undefined → 0 by our convention


def test_insertion_of_non_target_word_is_ignored(compute_tter):
    ref = "Laura met David"
    hyp = "Laura quickly met David"  # "quickly" inserted — not a target
    target_terms = {"Laura", "David"}
    result = compute_tter(ref, hyp, target_terms)
    assert result["insertions"] == 0


def test_empty_target_terms_returns_zero_tter(compute_tter):
    ref = "the cat sat on the mat"
    hyp = "the cat on mat"  # errors exist but nothing is a target
    target_terms = set()
    result = compute_tter(ref, hyp, target_terms)
    assert result["tter"] == 0.0
    assert result["target_total"] == 0


def test_case_insensitive_matching(compute_tter):
    ref = "Laura deployed to cloudflare"
    hyp = "Laura deployed to CloudFlare"  # case difference only
    target_terms = {"Cloudflare"}
    result = compute_tter(ref, hyp, target_terms)
    # Case-insensitive compare — should NOT count as substitution
    assert result["substitutions"] == 0
    assert result["tter"] == 0.0
