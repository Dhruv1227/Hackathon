"""Gazetteer-based place extraction with no Gemini calls.

Sources, in priority order when resolving a name:
  1. OSM features for the event region (neighbourhoods, rivers, bridges, roads, parks)
  2. CGNDB (official Canadian names), choosing the candidate nearest the region centre
  3. Hashtag / abbreviation aliases (#yyc -> Calgary, #highriverflood -> High River)

`python -m pipeline.gazetteer build` compiles the CGNDB subset into a compact file that ships
with the app (the raw 77 MB CSV is not needed at runtime).
"""
import csv
import gzip
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CGN_RAW = ROOT / "data/raw/cgn/cgn_canada_csv_eng.csv"
CGN_INDEX = ROOT / "data/gazetteer/cgndb_index.json.gz"
COMMON_WORDS = ROOT / "data/gazetteer/common_words.txt"

KEEP_GENERIC_CATEGORY = {"Populated Place"}
KEEP_TERMS = {
    "Indian Reserve": 0, "Indian Settlement": 0, "First Nation": 0, "Métis Settlement": 0,
    "City": 0, "Town": 0, "Village": 0, "Summer Village": 0, "Hamlet": 0,
    "Municipal District": 250000, "County": 250000, "Rural Municipality": 250000,
    "River": 250000, "Creek": 1000000, "Lake": 1000000, "Bridge": 0, "Dam": 0,
    "Provincial Park": 250000, "National Park": 0, "Reservoir": 250000,
}
# precision of a match, used for marker size on the map
PRECISION = {
    "neighbourhood": "point", "suburb": "point", "quarter": "point", "bridge": "point", "park": "point",
    "road": "line", "river": "line", "island": "point", "hamlet": "locality", "village": "locality",
    "locality": "locality", "town": "locality", "city": "locality", "first_nation": "locality",
    "province": "region", "region": "region",
}

PROVINCES = {
    "Alberta": (53.93, -116.58), "British Columbia": (53.73, -127.65), "Saskatchewan": (52.94, -106.45),
    "Manitoba": (53.76, -98.81), "Ontario": (51.25, -85.32), "Quebec": (52.94, -73.55),
    "New Brunswick": (46.57, -66.46), "Nova Scotia": (44.68, -63.74), "Prince Edward Island": (46.51, -63.42),
    "Newfoundland and Labrador": (53.14, -57.66), "Yukon": (64.28, -135.0), "Northwest Territories": (64.83, -124.85),
    "Nunavut": (70.3, -83.1),
}
# hashtags / abbreviations -> canonical names (airport codes are how Canadians tag cities)
ALIASES = {
    "yyc": "Calgary", "yeg": "Edmonton", "yql": "Lethbridge", "yxh": "Medicine Hat", "yqf": "Red Deer",
    "ymm": "Fort McMurray", "ywg": "Winnipeg", "yqt": "Thunder Bay", "yxe": "Saskatoon", "yqr": "Regina",
    "yvr": "Vancouver", "yyj": "Victoria", "ykf": "Kitchener", "yow": "Ottawa", "yyz": "Toronto", "yto": "Toronto",
    "yul": "Montreal", "ymq": "Montreal", "yqb": "Quebec City", "yhz": "Halifax", "yfc": "Fredericton",
    "yxy": "Whitehorse", "yzf": "Yellowknife", "yka": "Kamloops", "ylw": "Kelowna", "ypa": "Prince Albert",
    "ab": "Alberta", "alberta": "Alberta", "bc": "British Columbia", "sk": "Saskatchewan", "mb": "Manitoba",
    "on": "Ontario", "ont": "Ontario", "qc": "Quebec", "nb": "New Brunswick", "ns": "Nova Scotia",
    "pei": "Prince Edward Island", "nl": "Newfoundland and Labrador", "nwt": "Northwest Territories",
    "cowtown": "Calgary", "highriver": "High River", "medhat": "Medicine Hat", "kash": "Kashechewan", "mh": "Medicine Hat",
    # demonyms
    "calgarian": "Calgary", "calgarians": "Calgary", "albertan": "Alberta", "albertans": "Alberta",
    "edmontonian": "Edmonton", "edmontonians": "Edmonton", "winnipeggers": "Winnipeg", "manitobans": "Manitoba",
    "saskatchewanians": "Saskatchewan", "ontarians": "Ontario", "torontonians": "Toronto",
}
HASHTAG_SUFFIXES = ("floods", "flooding", "flood", "strong", "relief", "help", "helps", "news", "rain",
                    "storm", "evac", "evacuation", "water", "cleanup", "recovery", "proud", "love")
STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "is", "it", "we", "my", "our", "your",
    "this", "that", "all", "rt", "via", "amp", "new", "just", "now", "today", "tonight", "day", "canada",
    "city", "river", "lake", "downtown", "north", "south", "east", "west", "centre", "center", "park",
    "love", "strong", "help", "home", "stay", "safe", "high", "water", "flood", "bridge", "bow",
}
PREPS = {"in", "at", "near", "to", "from", "around", "across", "through", "into", "outside", "of", "@"}
# CGNDB names that are almost always organisations or ordinary words in tweets
BLOCK = {"red cross", "floods", "flood", "canada post", "global", "salvation army", "hwy", "highway",
         "downtown", "city hall", "emergency", "police", "fire", "news", "sun", "star", "herald",
         # companies / organisations that share names with places
         "shaw", "enmax", "telus", "rogers", "bell", "osler", "westjet", "safeway", "sobeys", "home depot",
         "tim hortons", "global", "cbc", "ctv", "city", "trends"}
BLOCK |= {"hurricane", "sandy", "typhoon", "tornado", "cyclone", "storm", "earthquake", "quake", "tsunami",
          "yolanda", "haiyan", "pablo", "bopha", "oswald", "marathon", "metro"}
# state/country names that are also everyday words and too vague to map on their own
GENERIC_ADMIN_WORDS = {"central", "western", "eastern", "northern", "southern", "capital", "coast", "coastal", "island",
                       "islands", "lakes", "mountains", "mountain", "valley", "border", "national", "federal", "union",
                       "plains", "highlands", "upper", "lower", "middle", "greater", "new", "north", "south", "east",
                       "west", "centre", "center", "interior", "orange", "delta", "lagoon", "river", "lake"}
# a name right after these words is a storm's name, not a place ("Hurricane Sandy" is not Sandy, Utah)
STORM_WORDS = {"hurricane", "superstorm", "typhoon", "storm", "cyclone", "tropical", "tornado", "ex-tropical"}
BLOCK_NOSPACE = {b.replace(" ", "") for b in BLOCK}
# names too broad to be useful as map points (Gemini sometimes returns them)
TOO_BROAD = {"canada", "north america", "world", "rockies", "rocky mountains", "western canada", "eastern canada",
             "the prairies", "prairies", "first nations", "usa", "united states"}
REGION_WORD_RE = re.compile(r"^(southern|northern|central|eastern|western|south|north|east|west)\s+(.+)$")
PREFIXES = ("downtown ", "the ", "city of ", "town of ", "village of ", "county of ")
SUFFIXES = (" first nation", " cree nation", " nation", " indian reserve", " reserve", " community", " neighbourhood",
            " neighborhood", " area", " district")


def fold(s: str) -> str:
    """Lowercase and strip accents: "Lac-Mégantic" -> "lac-megantic"."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def squash(s: str) -> str:
    """Letters and digits only: "Tsuu T'ina" and "Tsuut'ina" both -> "tsuutina"."""
    return re.sub(r"[^0-9a-z]", "", fold(s).replace("’", "'"))


def name_variants(key: str):
    """Progressively simplified forms of a place name, most specific first."""
    k = key.lower().replace("’", "'").strip()
    seen = []
    def add(v):
        v = v.strip()
        if v and v not in seen:
            seen.append(v)
    add(k)
    add(re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", k))  # 17th avenue -> 17 avenue (OSM style)
    for v in list(seen):
        for pre in PREFIXES:
            if v.startswith(pre):
                add(v[len(pre):])
    for v in list(seen):
        w = v
        for _ in range(3):  # "tsuut'ina nation 145" -> "tsuut'ina nation" -> "tsuut'ina"
            w2 = re.sub(r"\s+\d+[a-z]?$", "", w)
            for suf in SUFFIXES:
                if w2.endswith(suf):
                    w2 = w2[: -len(suf)]
            if w2 == w:
                break
            add(w2)
            w = w2
    return seen


QUADRANT_RE = re.compile(r"\s+(NE|NW|SE|SW|N|S|E|W)$")
TOKEN_RE = re.compile(r"#?[A-Za-zÀ-ÿ0-9'’\-\.]+")


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


# ------------------------------------------------------------------------ build step

def build():
    idx: dict[str, list] = {}
    kind_map = {"Indian Reserve": "first_nation", "Indian Settlement": "first_nation", "First Nation": "first_nation",
                "Métis Settlement": "first_nation", "River": "river", "Creek": "river", "Bridge": "bridge",
                "Dam": "bridge", "Provincial Park": "park", "National Park": "park", "Lake": "region",
                "Reservoir": "region", "Municipal District": "region", "County": "region", "Rural Municipality": "region",
                "City": "city", "Town": "town", "Village": "village", "Summer Village": "village", "Hamlet": "hamlet"}
    with CGN_RAW.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            term, cat = r["Generic Term"], r["Generic Category"]
            rel = int(r["Relevance at Scale"] or 0)
            if term in KEEP_TERMS:
                if rel < KEEP_TERMS[term]:
                    continue
            elif cat not in KEEP_GENERIC_CATEGORY:
                continue
            name = r["Geographical Name"]
            kind = kind_map.get(term, "locality")
            rec = [name, round(float(r["Latitude"]), 5), round(float(r["Longitude"]), 5),
                   r["Province - Territory"], kind, rel]
            idx.setdefault(name.lower(), []).append(rec)
            # "Siksika 146" -> also "Siksika"; "Kashechewan First Nation" -> "Kashechewan"
            short = re.sub(r"\s+(\d+[A-Z]?|First Nation|Indian Reserve|No\. ?\d+)$", "", name)
            if short != name and len(short) > 3:
                idx.setdefault(short.lower(), []).append(rec)
    CGN_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CGN_INDEX, "wt") as f:
        json.dump(idx, f, separators=(",", ":"))
    words = sorted({w.strip() for w in Path("/usr/share/dict/words").read_text().split() if w.islower()})
    COMMON_WORDS.write_text("\n".join(words))
    print(f"cgndb names={len(idx)} records={sum(map(len, idx.values()))} -> {CGN_INDEX} "
          f"({CGN_INDEX.stat().st_size / 1e6:.1f} MB); common words={len(words)}")


@lru_cache(maxsize=1)
def load_cgndb():
    with gzip.open(CGN_INDEX, "rt") as f:
        idx = json.load(f)
    nospace = {}
    for key, recs in list(idx.items()):
        if fold(key) != key:
            idx.setdefault(fold(key), recs)
        for v in name_variants(key)[1:]:  # short forms: "tsuut'ina nation 145" -> "tsuut'ina"
            if len(v) > 3:
                idx.setdefault(v, recs)
        if max(r[5] for r in recs) >= 250000 or any(r[4] == "first_nation" for r in recs):
            for v in name_variants(key):
                nospace.setdefault(squash(v), key)
    return idx, nospace


@lru_cache(maxsize=1)
def load_common_words():
    return set(COMMON_WORDS.read_text().split()) if COMMON_WORDS.exists() else set()


# ------------------------------------------------------------------------ runtime

class Gazetteer:
    def __init__(self, center=None, osm: dict | None = None, max_km: float = 600, canada: bool = True, anchor=None,
                 world: bool = False, region_cc: str | None = None):
        # world=True adds GeoNames cities/states/countries (needed for events outside Canada and global datasets)
        from pipeline.world import load_world
        self.world = load_world() if world else {}
        self.world_nospace = {}
        for key, recs in self.world.items():
            if " " in key and any(r[6] != "city" or r[5] >= 100_000 for r in recs):
                self.world_nospace.setdefault(squash(key), key)
        self.region_cc = region_cc or ("CA" if canada else None)
        # roads/rivers have many points: pick the one nearest the most-mentioned place (e.g. downtown Calgary),
        # not the region centroid, which can sit between two cities
        self.anchor = anchor or center
        self.canada = canada  # False for events outside Canada: CGNDB and Canadian aliases are skipped
        self.cgn, self.cgn_nospace = load_cgndb() if canada else ({}, {})
        self.common = load_common_words()
        self.center = center
        self.max_km = max_km
        self.osm = {}
        self.osm_nospace = {}
        for name, v in (osm or {}).items():
            base = QUADRANT_RE.sub("", name)
            keys = {name.lower(), base.lower()} | set(name_variants(base)[1:])
            for key in keys:
                if len(key) < 4 or key in STOP:
                    continue
                e = self.osm.setdefault(key, {"name": QUADRANT_RE.sub("", name), "kind": v["kind"], "points": []})
                e["points"].extend(v["points"])
                if " " in key or "'" in key:
                    self.osm_nospace.setdefault(squash(key), key)
        self.max_ngram = 5

    # ---- resolution
    def resolve(self, key: str, cc: str | None = None):
        """Resolve a name to a place dict, or None. Tries simplified variants ("Siksika First Nation" -> "Siksika",
        "Downtown Calgary" -> "Calgary", "17th Avenue" -> "17 Avenue") and region words ("Southern Alberta")."""
        key = key.lower().strip().replace("’", "'")
        if key in TOO_BROAD:
            return None
        for v in name_variants(key):
            p = self._resolve_one(v, cc)
            if p:
                return p
        m = REGION_WORD_RE.match(key)
        if m:
            p = self._resolve_one(m.group(2), cc)
            if p and p["precision"] == "region":
                return p
        k = (self.osm_nospace.get(squash(key)) or self.cgn_nospace.get(squash(key))
             or self.world_nospace.get(squash(key)))
        return self._resolve_one(k, cc) if k else None

    def _resolve_one(self, key: str, cc: str | None = None):
        from pipeline.world import DEMONYMS, REGION_ALIASES, load_countries
        key = key.lower().strip()
        if self.canada and key in ALIASES:
            key = ALIASES[key].lower()
        if self.world:
            if key in REGION_ALIASES:
                name, acc = REGION_ALIASES[key]
                return self._from_world(name.lower(), acc)
            if key in DEMONYMS and DEMONYMS[key] in load_countries():
                c = load_countries()[DEMONYMS[key]]
                return {"name": c["name"], "lat": c["lat"], "lon": c["lon"], "kind": "country", "source": "geonames",
                        "precision": "region", "province": None, "cc": DEMONYMS[key]}
        if self.canada and key in (p.lower() for p in PROVINCES):
            name = next(p for p in PROVINCES if p.lower() == key)
            lat, lon = PROVINCES[name]
            return {"name": name, "lat": lat, "lon": lon, "kind": "province", "source": "alias",
                    "precision": "region", "province": name, "cc": "CA"}
        o = self.osm.get(key) or self.osm.get(fold(key))
        if o and self.anchor:
            lat, lon = min(o["points"], key=lambda p: haversine(p, self.anchor))
            return {"name": o["name"], "lat": lat, "lon": lon, "kind": o["kind"], "source": "osm",
                    "precision": PRECISION.get(o["kind"], "point"), "province": None, "cc": self.region_cc}
        cands = self.cgn.get(key) or self.cgn.get(fold(key))
        if cands and self.center and (cc in (None, "CA")):
            def score(r):
                d = haversine((r[1], r[2]), self.center)
                return d - 60 * math.log10(max(r[5], 1))
            best = min(cands, key=score)
            d = haversine((best[1], best[2]), self.center)
            if d <= self.max_km or best[5] >= 7_500_000:
                return self._cgn_place(best)
        if self.world:
            p = self._from_world(fold(key), cc)
            if p:
                return p
        if cands and not self.center and cc in (None, "CA"):
            # no event centre (global data): only significant Canadian places and First Nations
            ok = [r for r in cands if r[5] >= 1_000_000 or r[4] == "first_nation"] if self.world else cands
            if ok:
                return self._cgn_place(max(ok, key=lambda r: r[5]))
        return None

    @staticmethod
    def _cgn_place(r):
        return {"name": r[0], "lat": r[1], "lon": r[2], "kind": r[4], "source": "cgndb",
                "precision": PRECISION.get(r[4], "locality"), "province": r[3], "rel": r[5], "cc": "CA"}

    def _from_world(self, key: str, cc: str | None = None):
        cands = self.world.get(key)
        if not cands:
            return None
        primary = [r for r in cands if not (len(r) > 7 and r[7])]
        if primary:
            cands = primary  # a town's real name beats another city's historical alias
        if cc:
            same = [r for r in cands if r[3] == cc.upper()]
            if same:
                cands = same
        if self.center:  # single event: nearest plausible candidate, big places may be far away
            def score(r):
                pop = r[5] if r[6] == "city" else 2_000_000
                return haversine((r[1], r[2]), self.center) - 60 * math.log10(max(pop, 1000))
            best = min(cands, key=score)
            far = haversine((best[1], best[2]), self.center) > self.max_km
            if far and best[6] == "city" and best[5] < 1_000_000:
                return None
        else:  # global: countries and states outrank same-named towns; otherwise the biggest city
            rank = {"country": 3, "admin1": 2, "city": 1}
            best = max(cands, key=lambda r: (rank[r[6]] if r[6] != "city" or r[5] < 1_000_000 else 2.5, r[5]))
        kind = {"country": "country", "admin1": "state", "city": "city"}[best[6]]
        return {"name": best[0], "lat": best[1], "lon": best[2], "kind": kind, "source": "geonames",
                "precision": "region" if best[6] != "city" else "locality", "province": best[4] or None,
                "pop": best[5], "cc": best[3]}

    def _hashtag_keys(self, tag: str):
        t = tag.lower().replace("_", "")
        yield t
        for suf in HASHTAG_SUFFIXES:
            if t.endswith(suf) and len(t) > len(suf) + 1:
                yield t[: -len(suf)]
        for pre in ("prayfor", "pray4", "helpthe", "help", "savethe"):
            if t.startswith(pre) and len(t) > len(pre) + 2:
                yield t[len(pre):]

    def _resolve_hashtag(self, tag: str):
        for k in self._hashtag_keys(tag):
            if k in BLOCK or k in BLOCK_NOSPACE:
                continue
            from pipeline.world import REGION_ALIASES
            if ((self.canada and k in ALIASES) or k in self.osm or k in self.cgn
                    or (self.world and (k in self.world or k in REGION_ALIASES))):
                if k in STOP or (len(k) < 4 and k not in ALIASES):
                    continue
                p = self.resolve(k)
                if p:
                    return {**p, "tag_key": k}
            key = self.osm_nospace.get(k) or self.cgn_nospace.get(k) or self.world_nospace.get(k)
            if key:
                p = self.resolve(key)
                if p:
                    return {**p, "tag_key": k}
        return None

    # ---- extraction
    def extract(self, text: str) -> list[dict]:
        toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        out, seen, i = [], set(), 0
        while i < len(toks):
            tok, s, e = toks[i]
            if tok.startswith("#"):
                p = self._resolve_hashtag(tok[1:])
                if p and p["name"] not in seen:
                    seen.add(p["name"])
                    out.append({**p, "surface": tok, "start": s, "end": e})
                i += 1
                continue
            matched = False
            for n in range(min(self.max_ngram, len(toks) - i), 0, -1):
                span = toks[i:i + n]
                if any(t[0].startswith("#") for t in span):
                    continue
                words = [t[0].strip(".'’-") for t in span]
                surface = " ".join(words)
                key = surface.lower()
                if key in BLOCK or not self._plausible(words, i):
                    continue
                if key in ALIASES and n == 1 and len(key) <= 3 and not words[0].isupper():
                    continue  # "on", "ab", "bc" as ordinary words
                p = self._resolve_one(key)  # exact lookup: this runs for every word span, so no variants here
                if p and i > 0 and toks[i - 1][0].lower().lstrip("#") in STORM_WORDS:
                    p = None  # "Hurricane Sandy", "Typhoon Pablo": a storm's name, not a place
                if p and n == 1 and p["source"] == "geonames" and p["kind"] == "city" and p.get("pop", 0) < 1_000_000 \
                        and i > 0:
                    prev = toks[i - 1][0]
                    if prev[:1].isupper() and prev.lower() not in PREPS and not self._resolve_one(prev.lower()):
                        p = None  # "Jeff Moore": a surname that happens to be a town
                if p and n == 1 and p["source"] == "cgndb" and p.get("rel", 0) < 1_000_000 and p["kind"] in ("locality", "hamlet") \
                        and not any(r[4] == "first_nation" for r in self.cgn.get(key, [])):
                    prev = toks[i - 1][0].lower() if i > 0 else ""
                    if prev not in PREPS:
                        p = None  # small one-word places ("Wilson", "Bruce") are usually people
                if p and n == 1 and p["source"] == "cgndb" and i > 0:
                    prev = toks[i - 1][0]
                    if prev[:1].isupper() and prev.lower() not in PREPS and not self._resolve_one(prev.lower()):
                        p = None  # "Jeff Wilson": preceded by a capitalised non-place word
                if p:
                    if p["name"] not in seen:
                        seen.add(p["name"])
                        out.append({**p, "surface": surface, "start": span[0][1], "end": span[-1][2]})
                    i += n
                    matched = True
                    break
            if not matched:
                i += 1
        return self._consistent(out) if self.world else out

    @staticmethod
    def _consistent(places: list[dict]) -> list[dict]:
        """When a tweet's other places agree on a country, drop small towns from elsewhere
        ("Boulder, Colorado; Lyons" -> not Lyon, France)."""
        from collections import Counter
        votes = Counter()
        for p in places:
            if p.get("cc"):  # a state/country mention is stronger evidence than one town
                votes[p["cc"]] += 2 if p["kind"] in ("country", "state", "province") else 1
        if not votes:
            return places
        top, n = votes.most_common(1)[0]
        if n < 2:
            return places
        return [p for p in places if p.get("cc") in (top, None) or p["kind"] in ("country", "state", "province")
                or p.get("pop", 0) >= 1_000_000]

    def _plausible(self, words: list[str], pos: int) -> bool:
        if not words[0] or not words[-1]:
            return False
        if words[0].lower() in STOP and len(words) == 1:
            return False
        cap = all(w[:1].isupper() or w[:1].isdigit() or w.lower() in ("of", "the", "de", "du", "la")
                  for w in words)
        if len(words) == 1:
            w = words[0]
            if len(w) < 3:
                from pipeline.world import REGION_ALIASES
                return (w.lower() in ALIASES or (self.world and w.lower() in REGION_ALIASES)) and w.isupper()
            if not w[:1].isupper():
                lw = w.lower()
                if lw in self.common or len(lw) < 5:
                    return False
                o = self.osm.get(lw)
                return bool(o and o["kind"] in ("town", "city", "village")) or \
                    any(r[5] >= 1_000_000 and r[4] in ("town", "city") for r in self.cgn.get(lw, [])) or \
                    any(r[6] == "city" and r[5] >= 500_000 for r in self.world.get(fold(lw), []))  # "manila", "boston"
            lw = w.lower()
            if lw in self.common or (lw.endswith("s") and lw[:-1] in self.common):
                # dictionary words that are real places: states/countries ("Colorado") anywhere, towns ("Mission",
                # "Boulder") only mid-sentence and only as in-region OSM names or sizeable cities
                recs = self.world.get(lw, [])
                if lw not in GENERIC_ADMIN_WORDS and any(r[6] in ("admin1", "country") for r in recs):
                    return True
                return pos > 0 and (lw in self.osm or any(r[6] == "city" and r[5] >= 100_000 for r in recs))
            return True
        return cap

    def mask(self, text: str, token: str = "PLACE") -> str:
        """Replace place mentions with a placeholder (used to train the location-agnostic model)."""
        spans = sorted(((m["start"], m["end"], m["surface"], m.get("tag_key")) for m in self.extract(text)),
                       reverse=True)
        for s, e, surf, tag_key in spans:
            if surf.startswith("#"):
                low = surf[1:].lower()
                j = low.find(tag_key) if tag_key else -1
                rep = "#" + (low[:j] + token + low[j + len(tag_key):] if j >= 0 else token)
            else:
                rep = token
            text = text[:s] + rep + text[e:]
        return text


# ------------------------------------------------------------------------ region detection

def detect_region(texts: list[str], top: int = 5):
    """Guess the event region from the most-mentioned significant places (no centre bias)."""
    g = Gazetteer(center=None)
    counts = Counter()
    places = {}
    for t in texts:
        for p in g.extract(t):
            if p["precision"] == "locality" and p["source"] != "osm":
                counts[p["name"]] += 1
                places[p["name"]] = p
    if not counts:
        return None
    best = [places[n] for n, _ in counts.most_common(top)]
    w = [counts[p["name"]] for p in best]
    # weight toward the dominant place so one distant mention doesn't drag the centre
    lat = sum(p["lat"] * c for p, c in zip(best, w)) / sum(w)
    lon = sum(p["lon"] * c for p, c in zip(best, w)) / sum(w)
    anchor = best[0]
    if haversine((lat, lon), (anchor["lat"], anchor["lon"])) > 300:
        lat, lon = anchor["lat"], anchor["lon"]
    # bbox snapped to a 0.25 degree grid so nearby uploads share the cached OSM download
    clat, clon = round(lat * 4) / 4, round(lon * 4) / 4
    return {"center": (round(lat, 4), round(lon, 4)), "top": counts.most_common(top),
            "anchor": (anchor["lat"], anchor["lon"]),
            "bbox": (clat - 1.0, clon - 1.75, clat + 1.0, clon + 1.75)}


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
