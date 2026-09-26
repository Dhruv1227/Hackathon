"""Per-dataset context: mode (one event vs worldwide), event region, gazetteer, and place masker.

Built once per upload. Works offline (OSM step is skipped if Overpass can't be reached).

Modes
  single, Canada   one event in Canada: CGNDB + OSM neighbourhoods for the event region (the main dataset)
  single, foreign  one event elsewhere (e.g. Colorado): GeoNames + OSM for the event region
  global           many events / countries (the bonus dataset): GeoNames + CGNDB, no single centre, world map
"""
import random
from collections import Counter
from dataclasses import dataclass, field

from pipeline.gazetteer import Gazetteer, detect_region, haversine
from pipeline.masking import Masker, distinctive_terms


@dataclass
class DatasetContext:
    region: dict | None
    gazetteer: Gazetteer
    masker: Masker
    terms: set = field(default_factory=set)

    @property
    def mode(self) -> str:
        return (self.region or {}).get("mode", "single")


def detect_mode(texts: list[str], sample: int = 8000, seed: int = 0) -> dict:
    """Count which countries a (sample of) tweets mention. One dominant country -> single event."""
    rnd = random.Random(seed)
    pool = texts if len(texts) <= sample else rnd.sample(texts, sample)
    g = Gazetteer(center=None, world=True)
    by_cc, places, recs = Counter(), Counter(), {}
    for t in pool:
        found = g.extract(t)
        for c in {p["cc"] for p in found if p.get("cc")}:
            by_cc[c] += 1
        for p in found:
            if p["precision"] == "locality" and p.get("cc"):
                places[(p["name"], p["cc"])] += 1
                recs[(p["name"], p["cc"])] = p
    total = sum(by_cc.values()) or 1
    shares = [(c, round(n / total, 3)) for c, n in by_cc.most_common(12)]
    top_cc, top_share = shares[0] if shares else (None, 0.0)
    n_big = sum(1 for _, sh in shares if sh >= 0.08)
    mode = "global" if shares and (top_share < 0.6 or n_big >= 3) else "single"
    return {"mode": mode, "top_cc": top_cc, "shares": shares, "places": places, "recs": recs}


def check_foreign(region: dict) -> dict:
    """If the most-mentioned place is, worldwide, clearly outside Canada (Boulder -> Colorado, not the
    hamlet in BC), re-centre on it and mark the event foreign. One cached Nominatim lookup."""
    from pipeline.gazetteer import load_cgndb
    from pipeline.geocode import geocode_global
    top = region["top"][0][0]
    cands = load_cgndb()[0].get(top.lower(), [])
    if max((r[5] for r in cands), default=0) >= 7_500_000:
        return region  # a major Canadian city: trust CGNDB
    try:
        g = geocode_global(top)
    except Exception:
        g = None
    if g and g["country_code"] and g["country_code"] != "ca":
        lat, lon = g["lat"], g["lon"]
        clat, clon = round(lat * 4) / 4, round(lon * 4) / 4
        return {**region, "center": (lat, lon), "anchor": (lat, lon),
                "bbox": (clat - 1.0, clon - 1.75, clat + 1.0, clon + 1.75),
                "country": g["country_code"].upper(), "foreign": True, "display": g["display"]}
    return region


def foreign_region(m: dict) -> dict | None:
    """Single event outside Canada, from the world-gazetteer counts of detect_mode."""
    top = [(k, n) for k, n in m["places"].most_common(10) if k[1] == m["top_cc"]]
    if not top:
        return None
    anchor = m["recs"][top[0][0]]
    near = [(m["recs"][k], n) for k, n in top
            if haversine((m["recs"][k]["lat"], m["recs"][k]["lon"]), (anchor["lat"], anchor["lon"])) < 300]
    w = sum(n for _, n in near)
    lat = sum(p["lat"] * n for p, n in near) / w
    lon = sum(p["lon"] * n for p, n in near) / w
    clat, clon = round(lat * 4) / 4, round(lon * 4) / 4
    return {"mode": "single", "center": (round(lat, 4), round(lon, 4)), "anchor": (anchor["lat"], anchor["lon"]),
            "bbox": (clat - 1.0, clon - 1.75, clat + 1.0, clon + 1.75), "country": m["top_cc"], "foreign": True,
            "top": [(k[0], n) for k, n in top[:5]]}


def build_context(texts: list[str], use_osm: bool = True) -> DatasetContext:
    m = detect_mode(texts)
    terms = distinctive_terms(texts)
    if m["mode"] == "global":
        region = {"mode": "global", "countries": m["shares"], "center": None, "country": None}
        g = Gazetteer(center=None, world=True)
        return DatasetContext(region=region, gazetteer=g, masker=Masker(g, terms), terms=terms)

    if m["top_cc"] in (None, "CA"):
        region = detect_region(texts)  # CGNDB-based centre: the main dataset's proven path
        if region and use_osm:
            region = check_foreign(region)
        if region:
            region = {"mode": "single", "country": "CA", **region}
    else:
        region = foreign_region(m)
    osm = {}
    if region and use_osm:
        try:
            from pipeline.osm import fetch
            osm = fetch(*region["bbox"])
        except Exception as e:  # network down, Overpass busy
            print(f"[context] OSM unavailable: {e}")
    foreign = bool((region or {}).get("foreign"))
    g = Gazetteer(center=region["center"] if region else None, osm=osm, canada=not foreign,
                  anchor=(region or {}).get("anchor"), world=True, region_cc=(region or {}).get("country"))
    return DatasetContext(region=region, gazetteer=g, masker=Masker(g, terms), terms=terms)
