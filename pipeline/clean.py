"""Clean tweets, detect retweets, and group copies so they never straddle a split.

Groups are built with union-find over three keys:
  1. normalized text (exact / near-exact duplicates)
  2. the text of an embedded retweet ("RT @user: ...") -> joins retweets and quote-RTs to the original
  3. a truncated retweet ("...…") joins any longer tweet sharing its normalized prefix
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
RT_PREFIX_RE = re.compile(r"^\s*RT\s+@\w+\s*:?\s*", re.I)
RT_EMBED_RE = re.compile(r"\bRT\s+@\w+\s*:?\s*(.+)$", re.I | re.S)
TRUNC_RE = re.compile(r"(?:…|\.\.\.)\s*$")
NON_WORD_RE = re.compile(r"[^\w#]+")
MIN_PREFIX = 40


def clean_text(text: str) -> str:
    """Human-readable clean text: unescape HTML, collapse whitespace."""
    text = html.unescape(str(text))
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    """Aggressive normalization for duplicate detection."""
    t = URL_RE.sub(" ", text.lower())
    t = RT_PREFIX_RE.sub("", t)
    t = MENTION_RE.sub(" ", t)
    t = TRUNC_RE.sub("", t)
    t = NON_WORD_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build(df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
    df = df.copy()
    df["row_id"] = range(len(df))
    df["text"] = df[text_col].fillna("").map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["is_retweet"] = df["text"].str.contains(r"\bRT\s+@\w+", regex=True)
    df["key"] = df["text"].map(norm_key)
    df["truncated"] = df["text"].str.contains(TRUNC_RE)

    n = len(df)
    uf = UnionFind(n)
    first_by_key = {}

    def link(key, i):
        if not key:
            return
        if key in first_by_key:
            uf.union(first_by_key[key], i)
        else:
            first_by_key[key] = i

    for i, (text, key) in enumerate(zip(df["text"], df["key"])):
        link(key, i)
        m = RT_EMBED_RE.search(text)
        if m:
            link(norm_key(m.group(1)), i)

    # truncated retweets: join to any tweet whose key starts with the truncated key
    keys_sorted = sorted(first_by_key)
    import bisect
    for i, (key, trunc) in enumerate(zip(df["key"], df["truncated"])):
        if not trunc or len(key) < MIN_PREFIX:
            continue
        # drop the final (possibly cut) word
        prefix = key.rsplit(" ", 1)[0]
        lo = bisect.bisect_left(keys_sorted, prefix)
        while lo < len(keys_sorted) and keys_sorted[lo].startswith(prefix):
            uf.union(first_by_key[keys_sorted[lo]], i)
            lo += 1

    df["group_id"] = [uf.find(i) for i in range(n)]
    df["group_size"] = df.groupby("group_id")["row_id"].transform("size")
    # canonical row per group: prefer a non-retweet, then the longest text
    df["_rank"] = (~df["is_retweet"]).astype(int) * 10000 + df["text"].str.len()
    canon = df.sort_values("_rank", ascending=False).drop_duplicates("group_id")
    df["is_canonical"] = df["row_id"].isin(canon["row_id"])
    return df.drop(columns=["_rank"])


def main(src="data/raw/main_contestant.csv", out="data/processed/clean.csv"):
    raw = pd.read_csv(src)
    df = build(raw)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    exact = raw["tweet"].duplicated().sum()
    print(f"rows={len(raw)} exact_dups={exact} retweets={df.is_retweet.sum()} "
          f"groups={df.group_id.nunique()} multi_row_groups={(df.drop_duplicates('group_id').group_size > 1).sum()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
