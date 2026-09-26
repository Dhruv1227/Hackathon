"""Group-aware split into tune (100), gold (200) and train pools.

- Splitting happens on group_id (after dedupe), so copies / retweets never cross pieces.
- Before hand labels exist we stratify on a keyword proxy for relevance so each labeled
  piece is roughly half relevant / half not. The real mix is checked after labeling.
- Only one canonical tweet per group is sampled into tune/gold.
"""
import re

import pandas as pd

SEED = 13
N_TUNE, N_GOLD = 100, 200
PROXY_RE = re.compile(
    r"flood|water|evac|river|\bbow\b|elbow|high ?river|canmore|sandbag|red ?cross|nenshi|"
    r"disaster|relief|donat|volunteer|power outage|boil|rescue|#yyc|#ab|shelter|emergency|stampede",
    re.I,
)


def main(src="data/processed/clean.csv", out="data/processed/clean.csv"):
    df = pd.read_csv(src)
    df["proxy_rel"] = df["text"].str.contains(PROXY_RE)
    groups = df[df["is_canonical"]][["group_id", "proxy_rel"]].sample(frac=1, random_state=SEED)

    def take(pool, n):
        pos = pool[pool.proxy_rel].head(n // 2)
        neg = pool[~pool.proxy_rel].head(n - len(pos))
        return pd.concat([pos, neg])["group_id"]

    gold = take(groups, N_GOLD)
    tune = take(groups[~groups.group_id.isin(gold)], N_TUNE)
    df["split"] = "train"
    df.loc[df.group_id.isin(tune), "split"] = "tune"
    df.loc[df.group_id.isin(gold), "split"] = "gold"
    df.to_csv(out, index=False)

    assert df.groupby("group_id")["split"].nunique().max() == 1, "group crosses splits"
    canon = df[df.is_canonical]
    print(canon.groupby("split")["proxy_rel"].agg(["size", "mean"]).round(2))
    print("proxy relevant share overall:", round(df.proxy_rel.mean(), 2))


if __name__ == "__main__":
    main()
