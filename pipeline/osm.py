"""Fetch named local features (neighbourhoods, rivers, bridges, parks, major roads) from
OpenStreetMap via Overpass for an event region, cached per bbox.

Each name keeps all its points (a river or road has many segments); the gazetteer
picks the point nearest to the region centre when resolving.
"""
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/cache/osm"
OVERPASS = "https://overpass-api.de/api/interpreter"
UA = "LivingFloodMap/0.1 (CE Strategies hackathon)"

QUERY = """[out:json][timeout:180];
(
  nwr["place"~"^(city|town|village|hamlet|suburb|neighbourhood|quarter|locality|island)$"]["name"]({b});
  way["waterway"~"^(river|stream|canal)$"]["name"]({b});
  way["bridge"="yes"]["name"]({b});
  way["man_made"="bridge"]["name"]({b});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]["name"]({b});
  relation["boundary"="administrative"]["admin_level"~"^(9|10)$"]["name"]({b});
  nwr["leisure"="park"]["name"]({b});
  nwr["boundary"="aboriginal_lands"]["name"]({b});
  nwr["leisure"~"^(stadium|sports_centre)$"]["name"]({b});
  nwr["amenity"~"^(school|college|university|hospital|community_centre|townhall)$"]["name"]({b});
  nwr["tourism"~"^(museum|attraction|zoo)$"]["name"]({b});
  way["landuse"~"^(retail|commercial)$"]["name"]({b});
  nwr["shop"="mall"]["name"]({b});
);
out center tags;"""

# Big rivers are often named only on a waterway relation whose member segments are unnamed (the Bow River
# through downtown Calgary). Fetch those members so the river snaps to the right stretch.
RIVER_QUERY = """[out:json][timeout:180];
relation["waterway"~"^(river|stream|canal)$"]["name"]({b})->.r;
way(r.r)({b})->.w;
.r out body;
.w out center;"""


def kind_of(tags: dict) -> str:
    if "place" in tags:
        return tags["place"]
    if tags.get("boundary") == "administrative":
        return "neighbourhood"
    if "waterway" in tags:
        return "river"
    if tags.get("bridge") == "yes" or tags.get("man_made") == "bridge":
        return "bridge"
    if "highway" in tags:
        return "road"
    if tags.get("leisure") == "park":
        return "park"
    if tags.get("boundary") == "aboriginal_lands":
        return "first_nation"
    if (tags.get("leisure") in ("stadium", "sports_centre") or "amenity" in tags or "tourism" in tags
            or "landuse" in tags or "shop" in tags):
        return "landmark"
    return "other"


def fetch(south: float, west: float, north: float, east: float) -> dict[str, dict]:
    """Return {name: {"kind": str, "points": [[lat, lon], ...]}} for the bbox."""
    bbox = ",".join(f"{v:.2f}" for v in (south, west, north, east))
    path = CACHE / f"{bbox}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(3):
        r = requests.post(OVERPASS, data={"data": QUERY.format(b=bbox)}, headers={"User-Agent": UA}, timeout=240)
        if r.status_code == 200:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    out: dict[str, dict] = {}
    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if not tags.get("name") or lat is None:
            continue
        # old/alt names matter: tweets use the name at the time (Langevin Bridge -> Reconciliation Bridge)
        names = [tags["name"]] + [n.strip() for k in ("old_name", "alt_name", "short_name")
                                  for n in tags.get(k, "").split(";") if n.strip()]
        if "/" in tags["name"]:  # "Bridgeland/Riverside" -> both communities
            names += [n.strip() for n in tags["name"].split("/") if len(n.strip()) > 3]
        for name in names:
            entry = out.setdefault(name, {"kind": kind_of(tags), "points": []})
            if len(entry["points"]) < 1500:  # long rivers/roads need their full length for snapping
                entry["points"].append([round(lat, 5), round(lon, 5)])
    try:
        _add_river_members(out, bbox)
    except Exception as e:  # optional extra; the main features are enough to work
        print(f"[osm] river relations skipped: {type(e).__name__}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def _add_river_members(out: dict, bbox: str):
    for attempt in range(3):  # Overpass throttles back-to-back queries (429/504): wait and retry
        time.sleep(5 + 10 * attempt)
        r = requests.post(OVERPASS, data={"data": RIVER_QUERY.format(b=bbox)}, headers={"User-Agent": UA}, timeout=240)
        if r.status_code == 200:
            break
    r.raise_for_status()
    els = r.json().get("elements", [])
    centers = {e["id"]: e["center"] for e in els if e["type"] == "way" and "center" in e}
    for rel in (e for e in els if e["type"] == "relation"):
        name = rel.get("tags", {}).get("name")
        if not name:
            continue
        entry = out.setdefault(name, {"kind": "river", "points": []})
        for m in rel.get("members", []):
            c = centers.get(m["ref"]) if m["type"] == "way" else None
            if c and len(entry["points"]) < 1500:
                entry["points"].append([round(c["lat"], 5), round(c["lon"], 5)])


if __name__ == "__main__":
    # Calgary + southern Alberta 2013 flood area (Canmore, High River, Okotoks, Siksika, Medicine Hat)
    feats = fetch(49.9, -115.6, 51.4, -110.5)
    from collections import Counter
    print(len(feats), Counter(v["kind"] for v in feats.values()).most_common())
    for n in ["Inglewood", "Bow River", "Elbow River", "Mission", "Bowness", "Sunnyside", "Blackfoot Trail", "Deerfoot Trail"]:
        print(n, feats.get(n, {}).get("kind"), (feats.get(n) or {}).get("points", [])[:2])
