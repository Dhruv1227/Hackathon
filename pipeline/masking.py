"""Place/event-name masking so the local model learns *flood language*, not Calgary names.

Two layers:
  1. Gazetteer matches (Canadian places, OSM region features, hashtag aliases).
  2. Event-distinctive proper nouns: capitalised words / hashtags that are far more frequent in
     this dataset than in a background corpus (e.g. "Boulder", "Sardinia", "Nenshi", "yycflood").
     This covers non-Canadian external events and names the gazetteer doesn't know.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKGROUND = ROOT / "data/gazetteer/background_df.json"
WORD_RE = re.compile(r"#?[A-Za-zÀ-ÿ][\w'’\-]*")
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
# words that are distinctive for floods in general and must stay visible to the model
KEEP = {"flood", "floods", "flooding", "flooded", "water", "rain", "river", "evacuation", "evacuated", "rescue",
        "help", "donate", "relief", "emergency", "red", "cross", "storm", "prayers", "damage", "power",
        "bridge", "road", "closed", "shelter", "volunteers", "safe", "stay", "update", "news", "alert"}
HASH_SUFFIX_RE = re.compile(r"(floods?|flooding|strong|relief|help|news|rain|storm)$", re.I)


def tokens(text: str) -> set[str]:
    text = MENTION_RE.sub(" ", URL_RE.sub(" ", text))
    return {w.lower().lstrip("#") for w in WORD_RE.findall(text)}


def doc_freq(texts) -> Counter:
    c = Counter()
    for t in texts:
        c.update(tokens(t))
    return c


def save_background(texts):
    df = doc_freq(texts)
    n = len(texts)
    BACKGROUND.parent.mkdir(parents=True, exist_ok=True)
    BACKGROUND.write_text(json.dumps({"n": n, "df": {k: v for k, v in df.items() if v >= 3}}))


def load_background():
    d = json.loads(BACKGROUND.read_text())
    return d["n"], d["df"]


def distinctive_terms(texts, background=None, min_share=0.004, ratio=8.0) -> set[str]:
    """Tokens appearing capitalised or as hashtags that are over-represented in `texts`."""
    n_bg, df_bg = background or load_background()
    cap = Counter()
    for t in texts:
        t = MENTION_RE.sub(" ", URL_RE.sub(" ", t))
        seen = set()
        for w in WORD_RE.findall(t):
            if w.startswith("#") or w[:1].isupper():
                seen.add(w.lower().lstrip("#"))
        cap.update(seen)
    n = len(texts)
    from pipeline.gazetteer import load_common_words
    common = load_common_words()
    out = set()
    for w, c in cap.items():
        share = c / n
        base = re.sub(r"['’]s$", "", w)
        if (share < min_share or len(w) < 3 or w in KEEP or "'" in base or "’" in base
                or base in common or base.rstrip("s") in common
                or (base.endswith("ing") and (base[:-3] in common or base[:-3] + "e" in common))):
            continue
        bg = (df_bg.get(w, 0) + 1) / (n_bg + 1)
        if share / bg >= ratio:
            out.add(w)
    return out


JUNK_RE = re.compile(r"[\u1f00-\u1fff\ufffd]+")  # mis-decoded emoji in HumAID


def mask_proper_nouns(text: str, common: set[str], token="PLACE") -> str:
    """Capitalised, non-sentence-initial, non-dictionary words -> token (for mixed-event corpora)."""
    out, prev_end = [], True
    for w in re.split(r"(\s+)", text):
        core = w.strip(".,!?:;\"'()").lstrip("#")
        if (core[:1].isupper() and not prev_end and core.lower() not in common
                and core.lower() not in KEEP and core not in ("PLACE", "URL", "RT", "I")
                and not core.startswith(("I'", "I’"))):
            w = w.replace(core, token)
        if w.strip():
            prev_end = w.rstrip()[-1:] in ".!?:"
        out.append(w)
    return "".join(out)


class Masker:
    def __init__(self, gazetteer=None, terms: set[str] | None = None, token="PLACE", proper_nouns=False):
        self.proper_nouns = proper_nouns
        self.g = gazetteer
        self.terms = terms or set()
        self.token = token
        self.term_re = None
        if self.terms:
            alt = "|".join(sorted(map(re.escape, self.terms), key=len, reverse=True))
            self.term_re = re.compile(rf"(?<![\w])(#?)({alt})(\w*)", re.I)

    def __call__(self, text: str) -> str:
        if self.g:
            text = self.g.mask(text, self.token)
        if self.term_re:
            def rep(m):
                return m.group(1) + self.token + m.group(3)
            text = self.term_re.sub(rep, text)
        text = MENTION_RE.sub("@user", URL_RE.sub("URL", JUNK_RE.sub("", text)))
        if self.proper_nouns:
            from pipeline.gazetteer import load_common_words
            text = mask_proper_nouns(text, load_common_words(), self.token)
        return re.sub(rf"({self.token}\s*)+", self.token + " ", text).strip()
