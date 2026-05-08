"""Unit tests for asr_correction.target_term_extractor."""

import pytest


@pytest.fixture(scope="module")
def extract():
    from asr_correction.target_term_extractor import extract_target_terms
    return extract_target_terms


def test_extracts_person_names(extract):
    text = "Laura is the project manager and David is the industrial designer."
    terms = extract(text)
    assert "Laura" in terms
    assert "David" in terms


def test_extracts_org_and_product_names(extract):
    text = "We deployed the new service on Cloudflare and Kubernetes last week."
    terms = extract(text)
    assert "Cloudflare" in terms
    assert "Kubernetes" in terms


def test_extracts_acronyms(extract):
    text = "The CVPR paper introduces a new CNN architecture for GPU inference."
    terms = extract(text)
    assert "CVPR" in terms
    assert "CNN" in terms
    assert "GPU" in terms


def test_extracts_technical_keywords(extract):
    text = "We apply the relu activation function before the softmax output."
    terms = extract(text)
    # technical keywords are case-insensitive, extractor returns the as-written form
    lowered = {t.lower() for t in terms}
    assert "relu" in lowered
    assert "softmax" in lowered


def test_filters_common_function_words(extract):
    text = "THE cat sat on THE mat AND looked FOR food."
    terms = extract(text)
    # THE, AND, FOR must not count as acronyms
    assert "THE" not in terms
    assert "AND" not in terms
    assert "FOR" not in terms


def test_filters_common_english_proper_noun_false_positives(extract):
    text = "On Monday morning, the report was due."
    terms = extract(text)
    # Day-of-week words are NER DATE, not PERSON/ORG/PRODUCT — should be excluded
    assert "Monday" not in terms


def test_empty_input_returns_empty_set(extract):
    assert extract("") == set()


def test_returns_a_set_not_a_list(extract):
    result = extract("Laura and David met at Cloudflare.")
    assert isinstance(result, set)


def test_all_caps_text_does_not_flood_with_acronyms(extract):
    # Simulates SlideAVSR ground truth format: all caps with occasional lowercase.
    text = (
        "WE WOULD LIKE TO PRESENT OUR PAPER PROGRESSIVE SEMANTIC SEGMENTATION "
        "IN CVPR two thousand and twenty-one. IMAGE SEMANTIC SEGMENTATION IS "
        "THE TASK WHERE THE MODEL APPLIES RELU AFTER EACH CONVOLUTION."
    )
    terms = extract(text)
    # Ordinary all-caps function words must NOT leak in as acronyms.
    flooded_words = {"WE", "WOULD", "LIKE", "TO", "PRESENT", "OUR", "THE",
                     "IS", "TASK", "WHERE", "MODEL", "APPLIES", "AFTER", "EACH"}
    leaked = terms & flooded_words
    assert not leaked, f"all-caps words leaked into terms: {leaked}"
