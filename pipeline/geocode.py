"""Fallback geocoder for place names the gazetteer can't resolve (usually Gemini-extracted
street addresses or businesses). Nominatim, biased to Canada and the event region, cached.
Respects the Nominatim usage policy: 1 request/second, identifying User-Agent, results cached.
"""
import json
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data/cache/geocode.json"
URL = "https://nominatim.openstreetmap.org/search"
UA = "LivingFloodMap/0.1 (CE Strategies hackathon)"
_lock = threading.Lock()
_last = [0.0]
_cache: dict | None = None


def _load():
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    return _cache


def _save():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache))


def geocode_global(name: str) -> dict | None:
    """Worldwide lookup of a single name; used once per upload to tell Canadian from foreign events."""
    key = f"GLOBAL|{name}"
    cache = _load()
    if key in cache:
        return cache[key]
    with _lock:
        wait = 1.05 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(URL, params={"q": name, "format": "jsonv2", "limit": 1, "addressdetails": 1},
                             headers={"User-Agent": UA}, timeout=10)
            _last[0] = time.time()
            hits = r.json() if r.ok else []
        except Exception:
            return None
    res = None
    if hits:
        h = hits[0]
        res = {"lat": float(h["lat"]), "lon": float(h["lon"]), "importance": h.get("importance", 0),
               "country_code": h.get("address", {}).get("country_code", ""), "display": h.get("display_name", "")}
    cache[key] = res
    _save()
    return res


def geocode(name: str, bbox=None, region_hint: str | None = None, country: str | None = "ca") -> dict | None:
    """bbox = (south, west, north, east) of the event region, used as a soft viewbox bias."""
    q = f"{name}, {region_hint}" if region_hint and region_hint.lower() not in name.lower() else name
    key = f"{q}|{bbox}|{country}"
    cache = _load()
    if key in cache:
        return cache[key]
    params = {"q": q, "format": "jsonv2", "limit": 1}
    if country:
        params["countrycodes"] = country
    if bbox:
        s, w, n, e = bbox
        params["viewbox"] = f"{w},{n},{e},{s}"
        params["bounded"] = 1
    with _lock:
        wait = 1.05 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(URL, params=params, headers={"User-Agent": UA}, timeout=10)
            _last[0] = time.time()
            hits = r.json() if r.ok else []
        except Exception:
            return None
    res = None
    if hits:
        h = hits[0]
        res = {"name": name, "lat": float(h["lat"]), "lon": float(h["lon"]), "kind": h.get("type", "place"),
               "source": "nominatim", "precision": "point", "province": None}
    cache[key] = res
    _save()
    return res
