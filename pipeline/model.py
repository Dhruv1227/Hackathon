"""Local multi-task classifier: relevance, category, urgency.

Features = MiniLM embedding of the place-masked text + a few generic flags.
Each task is a logistic-regression head, so training takes seconds on CPU and
predictions come with calibrated-ish probabilities for the budgeted cascade.

Training sources (all place-masked):
  relevance: CrisisLexT26 flood events (not Alberta) + Gemini labels of the main train split
  category : CrisisLexT26 + HumAID + Gemini labels
  urgency  : Gemini labels (no public source has urgency)
"""
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from pipeline.embed import embed
from pipeline.taxonomy import CATEGORY_CODES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models/local_model.joblib"
KW_RE = re.compile(r"flood|evacu|rescue|donat|sandbag|shelter|stranded|power (?:is )?out|boil|volunteer|"
                   r"relief|emergency|water level|high water|submerged|washed out|closed|damage|victims|prayers",
                   re.I)


def flags(masked: list[str]) -> np.ndarray:
    return np.array([[bool(KW_RE.search(t)), "PLACE" in t, t.startswith("RT "), "URL" in t]
                     for t in masked], dtype=np.float32)


def featurize(masked: list[str], cache_name: str | None = None) -> np.ndarray:
    return np.hstack([embed(masked, cache_name), flags(masked)])


def sqrt_class_weights(y) -> dict:
    """Softer than 'balanced': rare classes are boosted by sqrt(max/count). With 'balanced', PROPERTY (which only
    Gemini labels) was predicted for 26% of tweets vs Gemini's 10%; sqrt gave the best macro-F1 in 5-fold CV."""
    vals, counts = np.unique(y, return_counts=True)
    return dict(zip(vals, np.sqrt(counts.max() / counts)))


class LocalModel:
    def __init__(self):
        self.rel = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
        self.cat = LogisticRegression(C=2.0, max_iter=3000)
        self.urg = None
        self.haz = None  # hazard type of a relevant tweet (FLOOD, STORM, QUAKE, ...)

    def fit(self, X_rel, y_rel, w_rel, X_cat, y_cat, w_cat, X_urg=None, y_urg=None, w_urg=None,
            X_haz=None, y_haz=None, w_haz=None):
        self.rel.fit(X_rel, y_rel, sample_weight=w_rel)
        self.cat.set_params(class_weight=sqrt_class_weights(y_cat))
        self.cat.fit(X_cat, y_cat, sample_weight=w_cat)
        if X_urg is not None and len(set(y_urg)) > 1:
            # a relevant tweet is at least "general info"; sqrt weights keep urgency 3 near Gemini's rate
            # (balanced: 7.5% predicted vs 2% true; none: 0% -> life-safety tweets never surfaced)
            y_urg = np.clip(np.asarray(y_urg), 1, 3)
            self.urg = LogisticRegression(C=1.0, max_iter=2000, class_weight=sqrt_class_weights(y_urg))
            self.urg.fit(X_urg, y_urg, sample_weight=w_urg)
        if X_haz is not None and len(set(y_haz)) > 1:
            self.haz = LogisticRegression(C=2.0, max_iter=3000, class_weight=sqrt_class_weights(y_haz))
            self.haz.fit(X_haz, y_haz, sample_weight=w_haz)
        return self

    def predict(self, X) -> dict:
        p_rel = self.rel.predict_proba(X)[:, 1]
        p_cat = self.cat.predict_proba(X)
        cats = self.cat.classes_[p_cat.argmax(1)]
        if self.urg is not None:
            p_urg = self.urg.predict_proba(X)
            urg = self.urg.classes_[p_urg.argmax(1)].astype(int)
            urg_conf = p_urg.max(1)
        else:  # heuristic fallback before Gemini labels exist
            urg = np.array([{"HUMAN": 3, "NEEDS": 3, "EVAC": 2, "INFRA": 2, "PROPERTY": 2,
                             "ADVISORY": 2}.get(c, 1) for c in cats])
            urg_conf = np.full(len(cats), 0.5)
        if getattr(self, "haz", None) is not None:
            p_haz = self.haz.predict_proba(X)
            hazard = self.haz.classes_[p_haz.argmax(1)]
            p_flood = p_rel * p_haz[:, list(self.haz.classes_).index("FLOOD")]
        else:  # older models: every relevant tweet is treated as flood-related
            hazard = np.full(len(p_rel), "FLOOD")
            p_flood = p_rel
        return {"p_relevant": p_rel, "category": cats, "cat_conf": p_cat.max(1),
                "urgency": urg, "urg_conf": urg_conf, "hazard": hazard, "p_flood": p_flood,
                # 1.0 = coin flip, 0.0 = certain; drives which tweets go to Gemini
                "uncertainty": 1 - np.abs(p_rel - 0.5) * 2,
                "flood_uncertainty": 1 - np.abs(p_flood - 0.5) * 2}

    def save(self, path=MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH) -> "LocalModel":
        return joblib.load(path)


assert set(CATEGORY_CODES) >= {"INFRA", "AID"}
