"""World gazetteer from GeoNames (cities with 15k+ people, provinces/states, countries).

  .venv/bin/python -m pipeline.world build     # needs data/raw/geonames/{cities15000.txt,countryInfo.txt,admin1CodesASCII.txt}

Output (shipped with the app, ~1-2 MB):
  data/gazetteer/world_index.json.gz   name -> [[name, lat, lon, cc, admin1, population, kind], ...]
  data/gazetteer/countries.json        cc -> {name, lat, lon, continent}
Province/state and country centres are population-weighted means of their cities (no shapes needed).
"""
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/geonames"
INDEX = ROOT / "data/gazetteer/world_index.json.gz"
COUNTRIES = ROOT / "data/gazetteer/countries.json"
ASCII_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]{3,40}$")

# common ways tweets name countries/regions that GeoNames spells differently
COUNTRY_ALIASES = {
    "usa": "US", "u.s.": "US", "u.s.a.": "US", "united states of america": "US", "america": "US",
    "uk": "GB", "britain": "GB", "great britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "philippines": "PH", "phillipines": "PH", "philipines": "PH", "philipinnes": "PH", "pilipinas": "PH",
    "russia": "RU", "south korea": "KR", "north korea": "KP", "vietnam": "VN", "czech republic": "CZ",
    "holland": "NL", "the netherlands": "NL", "uae": "AE", "iran": "IR", "syria": "SY", "bolivia": "BO",
    "venezuela": "VE", "tanzania": "TZ", "laos": "LA", "burma": "MM", "ivory coast": "CI", "taiwan": "TW",
    "aussie": "AU", "oz": "AU",
}
DEMONYMS = {
    "filipino": "PH", "filipinos": "PH", "pinoy": "PH", "australian": "AU", "australians": "AU", "aussies": "AU",
    "italian": "IT", "italians": "IT", "american": "US", "americans": "US", "canadian": "CA", "canadians": "CA",
    "mexican": "MX", "brazilian": "BR", "guatemalan": "GT", "venezuelan": "VE", "bangladeshi": "BD",
    "spanish": "ES", "russian": "RU", "russians": "RU", "singaporean": "SG", "singaporeans": "SG",
    "indonesian": "ID", "indian": "IN", "pakistani": "PK", "japanese": "JP", "chinese": "CN", "british": "GB",
}
# regional hashtags/abbreviations used in disaster tweets
REGION_ALIASES = {
    "qld": ("Queensland", "AU"), "nsw": ("New South Wales", "AU"), "vic": ("Victoria", "AU"),
    "ph": ("Philippines", "PH"), "pinas": ("Philippines", "PH"), "nyc": ("New York City", "US"),
    "socal": ("California", "US"), "norcal": ("California", "US"), "lax": ("Los Angeles", "US"),
    "sardegna": ("Sardinia", "IT"), "cdo": ("Cagayan de Oro", "PH"), "ncr": ("Manila", "PH"),
    "metro manila": ("Manila", "PH"), "mm": ("Manila", "PH"),
}


def build():
    countries, cc_name = {}, {}
    with (RAW / "countryInfo.txt").open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            cc_name[p[0]] = p[4]
            countries[p[0]] = {"name": p[4], "continent": p[8], "sx": 0.0, "sy": 0.0, "w": 0.0}
    admin1 = {}
    with (RAW / "admin1CodesASCII.txt").open(encoding="utf-8") as f:
        for line in f:
            code, name, ascii_name, _ = line.rstrip("\n").split("\t")
            admin1[code] = {"name": name, "ascii": ascii_name, "sx": 0.0, "sy": 0.0, "w": 0.0}

    idx = defaultdict(list)
    with (RAW / "cities15000.txt").open(encoding="utf-8") as f:
        for p in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            name, ascii_name, alts = p[1], p[2], p[3]
            lat, lon, cc, a1, pop = float(p[4]), float(p[5]), p[8], p[10], int(p[14] or 0)
            a1name = admin1.get(f"{cc}.{a1}", {}).get("name", "")
            rec = [name, round(lat, 4), round(lon, 4), cc, a1name, pop, "city", 0]
            keys = {name.lower(), ascii_name.lower()}
            for k in keys:
                idx[k].append(rec)
            if pop >= 200_000:  # alternate spellings only for big cities; flagged so real names win ("Olbia")
                alt_rec = rec[:7] + [1]
                for k in {a.lower() for a in alts.split(",") if ASCII_NAME_RE.match(a)} - keys:
                    idx[k].append(alt_rec)
            w = max(pop, 1) ** 0.5  # sqrt so one megacity doesn't pull the centre onto itself
            for acc in (countries.get(cc), admin1.get(f"{cc}.{a1}")):
                if acc is not None:
                    acc["sx"] += lat * w; acc["sy"] += lon * w; acc["w"] += w

    for code, a in admin1.items():
        if a["w"]:
            cc = code.split(".")[0]
            rec = [a["name"], round(a["sx"] / a["w"], 3), round(a["sy"] / a["w"], 3), cc, a["name"], 0, "admin1", 0]
            for k in {a["name"].lower(), a["ascii"].lower()}:
                idx[k].insert(0, rec)
    out_countries = {}
    for cc, c in countries.items():
        if not c["w"]:
            continue
        out_countries[cc] = {"name": c["name"], "lat": round(c["sx"] / c["w"], 3), "lon": round(c["sy"] / c["w"], 3),
                             "continent": c["continent"]}
        rec = [c["name"], out_countries[cc]["lat"], out_countries[cc]["lon"], cc, "", 0, "country", 0]
        idx[c["name"].lower()].insert(0, rec)
    for alias, cc in COUNTRY_ALIASES.items():
        if cc in out_countries:
            c = out_countries[cc]
            idx[alias].insert(0, [c["name"], c["lat"], c["lon"], cc, "", 0, "country", 0])

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(INDEX, "wt") as f:
        json.dump(idx, f, separators=(",", ":"))
    COUNTRIES.write_text(json.dumps(out_countries, separators=(",", ":")))
    print(f"world names={len(idx)} -> {INDEX} ({INDEX.stat().st_size / 1e6:.1f} MB); countries={len(out_countries)}")


@lru_cache(maxsize=1)
def load_world():
    with gzip.open(INDEX, "rt") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_countries() -> dict:
    return json.loads(COUNTRIES.read_text())


def country_name(cc: str | None) -> str:
    return load_countries().get((cc or "").upper(), {}).get("name", cc or "")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
