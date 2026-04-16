"""Curated technical and business terms for target term extraction.

These are common-English words that would NOT be caught by NER or proper-noun
rules but are still domain-relevant and should count as target terms for TTER.

Lower-cased; the extractor compares case-insensitively.
"""

TECHNICAL_KEYWORDS: frozenset[str] = frozenset({
    # AI / ML
    "transformer", "attention", "embedding", "gradient", "backprop",
    "encoder", "decoder", "tokenizer", "softmax", "relu", "sigmoid",
    "inference", "fine-tuning", "pretraining", "prompt", "token",
    "logits", "checkpoint", "hyperparameter", "convolution", "regression",
    "classification", "clustering", "perceptron", "quantization",

    # Software engineering
    "api", "endpoint", "schema", "payload", "microservice", "middleware",
    "kubernetes", "docker", "container", "pipeline", "webhook",
    "dependency", "framework", "runtime", "compiler", "bytecode",
    "database", "cache", "queue", "broker", "cluster", "deployment",

    # Business / finance
    "revenue", "margin", "ebitda", "capex", "opex", "valuation",
    "quarterly", "forecast", "dividend", "equity", "liquidity",
    "portfolio", "fiscal", "gaap", "shareholder", "earnings",
})
