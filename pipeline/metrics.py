"""Shared scoring helpers: place-name matching that tolerates typos and aliases."""
from rapidfuzz import fuzz

PROVINCES = {"alberta", "british columbia", "saskatchewan", "manitoba", "ontario", "quebec", "canada"}


def canon_place(name: str, gazetteer=None) -> str:
    n = name.strip()
    if gazetteer is not None:
        p = gazetteer.resolve(n.lower().lstrip("#"))
        if p:
            return p["name"].lower()
    return n.lower().lstrip("#")


def place_match(pred: list[str], gold: list[str], gazetteer=None, skip_regions=True, threshold=85):
    """Return (hits, n_gold, n_pred) with fuzzy + gazetteer-canonical matching.
    Province/country names are skipped by default: they're trivially right and would inflate the score."""
    g = [canon_place(x, gazetteer) for x in gold if x.strip()]
    p = [canon_place(x, gazetteer) for x in pred if x.strip()]
    if skip_regions:
        g = [x for x in g if x not in PROVINCES]
        p = [x for x in p if x not in PROVINCES]
    used, hits = set(), 0
    for x in g:
        for i, y in enumerate(p):
            if i not in used and (x == y or fuzz.ratio(x, y) >= threshold or fuzz.token_set_ratio(x, y) >= 95):
                used.add(i)
                hits += 1
                break
    return hits, len(g), len(p)
