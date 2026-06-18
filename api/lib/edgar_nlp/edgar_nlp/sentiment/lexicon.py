"""Loughran-McDonald lexicon — a small, FIXED reference word-list subset.

This ships a curated subset of the Loughran-McDonald Master Dictionary (2011)
finance-sentiment categories as plain frozensets. It is REFERENCE DATA, not a
fitted model: the words are looked up verbatim and never learned. The full
dictionary is ~85k entries across negative/positive/uncertainty/litigious/
strong-modal/weak-modal/constraining categories; this subset keeps the package
self-contained and offline while preserving the category structure the scorer
reports.

Provenance: Loughran, T. & McDonald, B. (2011), "When Is a Liability Not a
Liability? Textual Analysis, Dictionaries, and 10-Ks", Journal of Finance 66(1).
Words below are uppercase in the source dictionary; the scorer matches on
lowercased tokens, so they are stored lowercased here.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Final

# NOTE: A representative curated subset. The production vendor bundle may replace
# this with the full Master-Dictionary export; the scorer logic is identical.

#: Negative-sentiment words (the dominant, highest-signal LM category for 10-Ks).
LM_NEGATIVE: Final[frozenset[str]] = frozenset(
    {
        "loss",
        "losses",
        "decline",
        "declines",
        "declined",
        "declining",
        "adverse",
        "adversely",
        "litigation",
        "default",
        "defaulted",
        "bankruptcy",
        "bankruptcies",
        "impairment",
        "impairments",
        "deficiency",
        "deficiencies",
        "weakness",
        "weaknesses",
        "downturn",
        "lawsuit",
        "lawsuits",
        "negative",
        "negatively",
        "fail",
        "failed",
        "failure",
        "failures",
        "deteriorate",
        "deterioration",
        "shortfall",
        "shortfalls",
        "termination",
        "terminated",
        "restated",
        "restatement",
        "fraud",
        "penalties",
        "penalty",
        "doubt",
        "doubtful",
    }
)

#: Positive-sentiment words (a small category in 10-Ks; partly subsumed by tone).
LM_POSITIVE: Final[frozenset[str]] = frozenset(
    {
        "gain",
        "gains",
        "gained",
        "growth",
        "improve",
        "improved",
        "improvement",
        "improvements",
        "strong",
        "stronger",
        "strength",
        "profitable",
        "profitability",
        "favorable",
        "favorably",
        "success",
        "successful",
        "successfully",
        "benefit",
        "benefits",
        "beneficial",
        "efficient",
        "efficiencies",
        "opportunity",
        "opportunities",
        "leading",
        "outperform",
        "outperformed",
        "exceeded",
        "achievement",
        "achievements",
    }
)

#: Uncertainty words (modal/hedging; high in Risk Factors).
LM_UNCERTAINTY: Final[frozenset[str]] = frozenset(
    {
        "may",
        "could",
        "might",
        "uncertain",
        "uncertainty",
        "uncertainties",
        "risk",
        "risks",
        "approximate",
        "approximately",
        "assume",
        "assumed",
        "assumption",
        "assumptions",
        "depend",
        "depends",
        "dependent",
        "fluctuate",
        "fluctuates",
        "fluctuation",
        "fluctuations",
        "possible",
        "possibly",
        "predict",
        "predicted",
        "probable",
        "anticipate",
        "anticipated",
        "believe",
        "believes",
        "estimate",
        "estimates",
        "exposure",
        "indefinite",
        "tentative",
        "variable",
        "volatility",
    }
)

#: Litigious words (legal/regulatory exposure).
LM_LITIGIOUS: Final[frozenset[str]] = frozenset(
    {
        "claimant",
        "contract",
        "contractual",
        "court",
        "courts",
        "defendant",
        "plaintiff",
        "regulation",
        "regulations",
        "regulatory",
        "statute",
        "statutes",
        "statutory",
        "subpoena",
        "testimony",
        "tort",
        "judicial",
        "jurisdiction",
        "legislation",
        "liable",
        "liability",
        "liabilities",
        "indemnification",
        "settlement",
        "settlements",
        "compliance",
        "noncompliance",
        "enforcement",
    }
)

#: Constraining words (limits on managerial flexibility).
LM_CONSTRAINING: Final[frozenset[str]] = frozenset(
    {
        "require",
        "requires",
        "required",
        "requirement",
        "requirements",
        "obligation",
        "obligations",
        "obligated",
        "mandatory",
        "must",
        "restrict",
        "restricted",
        "restriction",
        "restrictions",
        "covenant",
        "covenants",
        "prohibit",
        "prohibited",
        "limit",
        "limited",
        "limitation",
        "limitations",
        "comply",
        "constrain",
        "constrained",
        "commitment",
        "commitments",
    }
)

#: The five reported categories, keyed by the slug used in the API/CLI response.
LM_CATEGORIES: Final[dict[str, frozenset[str]]] = {
    "negative": LM_NEGATIVE,
    "positive": LM_POSITIVE,
    "uncertainty": LM_UNCERTAINTY,
    "litigious": LM_LITIGIOUS,
    "constraining": LM_CONSTRAINING,
}
