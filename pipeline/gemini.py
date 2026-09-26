"""Budgeted Gemini client + compact batch labeling.

Every real API call is appended to a ledger (data/cache/gemini_ledger.jsonl) so the
number of calls spent is always known. Responses are cached by prompt hash, so
re-running a script never spends a call twice. With no GEMINI_API_KEY the client
runs in dry-run mode: it returns an empty response and records nothing.

Batch output format, one line per tweet:
    id|relevant|confidence|CATEGORY|urgency|loc1;loc2
    e.g. 12|1|9|INFRA|2|Inglewood;Blackfoot Trail
"""
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from pipeline.taxonomy import CATEGORIES, CATEGORY_CODES, HAZARD_CODES, HAZARDS

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
CACHE_DIR = ROOT / "data/cache/gemini"
LEDGER = ROOT / "data/cache/gemini_ledger.jsonl"
PROMPT_VERSION = "v3"


class BudgetExceeded(RuntimeError):
    pass


# Keep this many proxy requests in reserve: uploads stop using Gemini (local model only) below it.
QUOTA_RESERVE = int(os.getenv("GEMINI_QUOTA_RESERVE", "100"))


QUOTA_TOTAL = int(os.getenv("GEMINI_QUOTA_TOTAL", "1000"))


def estimated_remaining() -> int | None:
    """Conservative remaining quota: the lower of the proxy's own figure and (quota - requests we logged, including
    failed ones). The proxy under-counts parallel requests (20 concurrent requests moved its counter by 6), so we
    never rely on its number alone."""
    ours = None
    if LEDGER.exists():
        lines = LEDGER.read_text().splitlines()
        ours = QUOTA_TOTAL - len(lines)  # successes and failures both count
    vals = [v for v in (last_known_remaining(), ours) if v is not None]
    return min(vals) if vals else None


def last_known_remaining() -> int | None:
    """requests_remaining from the most recent successful call (ledger), if the proxy reports it."""
    if not LEDGER.exists():
        return None
    for line in reversed(LEDGER.read_text().splitlines()):
        d = json.loads(line)
        if d.get("requests_remaining") is not None:
            return d["requests_remaining"]
    return None


class APIError(RuntimeError):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


class Gemini:
    """Backends, chosen by GEMINI_BACKEND in .env (default: auto):
      hackathon  organisers' proxy: POST {GEMINI_BASE_URL}/api/generate, X-API-Key header,
                 {"contents", "model"?} -> {"text", "requests_remaining"}. Failed requests count
                 toward the quota, and 429 means the quota is exhausted, so retries are minimal.
      google  Google AI Studio key, or a Gemini-native proxy via GEMINI_BASE_URL
      openai  OpenAI-compatible proxy (LiteLLM etc.): POST {GEMINI_BASE_URL}/chat/completions
      vertex  Vertex AI: GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION (ADC), or an express-mode key
    auto = openai if GEMINI_BASE_URL looks OpenAI-style (/v1, /openai), google otherwise.
    """

    def __init__(self, budget: int | None = None, model: str | None = None, allow_reserve: bool = False):
        self.allow_reserve = allow_reserve  # summaries may dip into the reserve; bulk labeling may not
        self.cache_only = False
        self.key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_env = model or os.getenv("GEMINI_MODEL", "").strip()
        self.base_url = os.getenv("GEMINI_BASE_URL", "").strip().rstrip("/")
        self.backend = os.getenv("GEMINI_BACKEND", "auto").strip().lower()
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        if self.backend == "auto":
            if project and not self.base_url:
                self.backend = "vertex"
            elif self.base_url and re.search(r"/v1$|/v1/|openai", self.base_url):
                self.backend = "openai"
            elif self.base_url:
                self.backend = "hackathon"
            else:
                self.backend = "google"
        default_model = "gemini-3-flash-preview" if self.backend == "hackathon" else "gemini-2.5-flash"
        self.model = self.model_env or default_model
        self.requests_remaining = None
        self.budget = budget  # max requests for this client instance (None = unlimited), failed attempts included
        self.spent = 0
        self._lock = threading.Lock()
        self.dry_run = not (self.key or (self.backend == "vertex" and project))
        self._client = None
        if self.dry_run or self.backend in ("openai", "hackathon"):
            return
        from google import genai
        from google.genai import types
        if self.backend == "vertex":
            if project:
                self._client = genai.Client(vertexai=True, project=project,
                                            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
            else:  # Vertex express mode
                self._client = genai.Client(vertexai=True, api_key=self.key)
        else:
            http = types.HttpOptions(base_url=self.base_url) if self.base_url else None
            self._client = genai.Client(api_key=self.key, http_options=http)

    def _call(self, prompt: str, temperature: float):
        """Returns (text, input_tokens, output_tokens). Raises with a .code on HTTP errors."""
        if self.backend == "hackathon":
            import requests
            body = {"contents": prompt}
            if self.model_env:
                body["model"] = self.model_env
            try:
                r = requests.post(f"{self.base_url}/api/generate", timeout=300, json=body,
                                  headers={"X-API-Key": self.key, "Content-Type": "application/json"})
            except requests.RequestException as e:
                raise APIError(599, f"network: {type(e).__name__}") from e
            if r.status_code != 200:
                raise APIError(r.status_code, r.text[:300])
            d = r.json()
            self.requests_remaining = d.get("requests_remaining")
            return d.get("text") or "", None, None
        if self.backend == "openai":
            import requests
            header = os.getenv("GEMINI_AUTH_HEADER", "Authorization")
            auth = f"Bearer {self.key}" if header.lower() == "authorization" else self.key
            r = requests.post(f"{self.base_url}/chat/completions", timeout=300, headers={header: auth},
                              json={"model": self.model, "temperature": temperature,
                                    "messages": [{"role": "user", "content": prompt}]})
            if r.status_code != 200:
                raise APIError(r.status_code, r.text[:300])
            d = r.json()
            u = d.get("usage") or {}
            return d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens"), u.get("completion_tokens")
        from google.genai import types
        cfg = types.GenerateContentConfig(temperature=temperature,
                                          thinking_config=types.ThinkingConfig(thinking_budget=0))
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        u = getattr(resp, "usage_metadata", None)
        return resp.text or "", getattr(u, "prompt_token_count", None), getattr(u, "candidates_token_count", None)

    @staticmethod
    def ledger_total() -> int:
        """Successful calls. Failed attempts are logged separately (see ledger_failures)."""
        if not LEDGER.exists():
            return 0
        return sum(1 for line in LEDGER.open() if '"failed"' not in line)

    @staticmethod
    def ledger_failures() -> int:
        if not LEDGER.exists():
            return 0
        return sum(1 for line in LEDGER.open() if '"failed"' in line)

    def _reserve(self):
        """Claim one request under a lock, so parallel threads can't all pass the check and overshoot the cap."""
        with self._lock:
            if self.budget is not None and self.spent >= self.budget:
                raise BudgetExceeded(f"budget of {self.budget} calls used")
            remaining = estimated_remaining()
            if remaining is not None and (remaining <= 10 or (remaining <= QUOTA_RESERVE and not self.allow_reserve)):
                raise BudgetExceeded(f"only {remaining} proxy requests left (reserve {QUOTA_RESERVE})")
            self.spent += 1

    def _log_failure(self, tag, h, err, code):
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "tag": tag, "model": self.model,
                                "hash": h, "failed": True, "code": code, "error": type(err).__name__}) + "\n")

    def generate(self, prompt: str, tag: str = "", temperature: float = 0.0) -> str:
        h = hashlib.sha256(f"{self.model}\n{temperature}\n{prompt}".encode()).hexdigest()[:24]
        cached = CACHE_DIR / f"{h}.txt"
        if cached.exists():
            return cached.read_text()
        if self.dry_run:
            print(f"[gemini dry-run] {tag}: {len(prompt)} chars, not sent")
            return ""
        if self.cache_only:  # replay mode: answers must come from the cache, never from the network
            raise BudgetExceeded("cache-only replay: prompt not cached")
        # the hackathon proxy counts failed requests and uses 429 for "quota exhausted": retry 5xx once, never 4xx
        max_attempts = 2 if self.backend == "hackathon" else 4
        for attempt in range(max_attempts):
            self._reserve()  # every attempt is one request against the budget, checked atomically across threads
            t0 = time.time()
            try:
                text, n_in, n_out = self._call(prompt, temperature)
                break
            except Exception as e:
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                self._log_failure(tag, h, e, code)
                client_error = isinstance(code, int) and 400 <= code < 500
                if attempt == max_attempts - 1 or (client_error and (code != 429 or self.backend == "hackathon")):
                    raise
                wait = 2 ** attempt * 5
                print(f"[gemini] {type(e).__name__} {code} -- retrying in {wait}s")
                time.sleep(wait)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(text)
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(), "tag": tag, "model": self.model, "backend": self.backend,
                "hash": h, "secs": round(time.time() - t0, 1), "in_tokens": n_in, "out_tokens": n_out,
                "requests_remaining": self.requests_remaining,
            }) + "\n")
        return text


# ---------------------------------------------------------------- batch labeling

def label_prompt(tweets: list[str], event_hint: str | None = None) -> str:
    cats = "\n".join(f"  {k}: {v}" for k, v in CATEGORIES.items())
    event = (f"The tweets were collected during this event: {event_hint}.\n" if event_hint else
             "The tweets were collected during a flood disaster; infer the event from the tweets.\n")
    lines = "\n".join(f"{i}\t{t.replace(chr(10), ' ')}" for i, t in enumerate(tweets))
    return f"""You label tweets for a flood-response map. {event}
For EVERY tweet output exactly one line, no header, no extra text.
Relevant tweet:     id|1|confidence|CATEGORY|urgency|locations
Not relevant tweet: id|0|confidence

relevant: 1 if the tweet is about the flood disaster (conditions, impacts, response, aid, advice,
  sympathy, commentary on the flood), else 0. Unrelated personal chatter, ads, sports, other news = 0.
confidence: 1-9, how sure you are about `relevant` (9 = certain).
CATEGORY (pick the single best fit):
{cats}
urgency: 0 none, 1 general info, 2 active impact / disruption,
  3 immediate threat to life or safety / urgent request.
locations: specific places named or clearly implied in the tweet (neighbourhoods, streets,
  bridges, rivers, parks, towns, First Nations), separated by ';'. Expand hashtags/abbreviations
  to the place name (#yyc -> Calgary). Use - if none. Do not invent places.

Examples:
12|1|9|INFRA|2|Inglewood;Blackfoot Trail
13|0|8

Tweets (id<TAB>text):
{lines}
"""


def label_prompt_v4(tweets: list[str], event_hint: str | None = None) -> str:
    """v4 adds the hazard type and country, for multi-disaster / world datasets and uploads.
    (v3 above stays frozen: it is the prompt the gold evaluation was run with.)"""
    cats = "\n".join(f"  {k}: {v}" for k, v in CATEGORIES.items())
    hz = "\n".join(f"  {k}: {v}" for k, v in HAZARDS.items())
    event = (f"The tweets were collected around: {event_hint}.\n" if event_hint else
             "The tweets may come from many different disasters around the world.\n")
    lines = "\n".join(f"{i}\t{t.replace(chr(10), ' ')}" for i, t in enumerate(tweets))
    return f"""You label tweets for a flood-monitoring world map. {event}
For EVERY tweet output exactly one line, no header, no extra text.
Relevant tweet:     id|1|confidence|HAZARD|CATEGORY|urgency|country|locations
Not relevant tweet: id|0|confidence

relevant: 1 if the tweet is about a real disaster or emergency (conditions, impacts, response, aid, advice,
  sympathy, commentary on it), else 0. Personal chatter, ads, sports, jokes, unrelated news = 0.
confidence: 1-9, how sure you are about `relevant` (9 = certain).
HAZARD (what this tweet itself is about):
{hz}
CATEGORY (impact type, single best fit):
{cats}
urgency: 0 none, 1 general info, 2 active impact / disruption,
  3 immediate threat to life or safety / urgent request.
country: ISO 3166-1 alpha-2 code of where the event is (e.g. CA, US, PH, AU, IT); - if unclear.
locations: specific places named or clearly implied in the tweet (neighbourhoods, streets, bridges, rivers,
  towns, provinces, First Nations), separated by ';'. Expand hashtags/abbreviations to the place name
  (#yyc -> Calgary, #qldfloods -> Queensland). Use - if none. Do not invent places.

Examples:
12|1|9|FLOOD|INFRA|2|CA|Inglewood;Blackfoot Trail
13|0|8
14|1|8|QUAKE|HUMAN|3|CR|Nicoya

Tweets (id<TAB>text):
{lines}
"""


def _parse_line(line: str):
    """Tolerant parse: id|rel|conf[|CAT|urg|locs]. Extra '-' fields and stray backticks are ignored."""
    parts = [p.strip() for p in line.strip().strip("`").split("|")]
    if len(parts) < 3 or not parts[0].isdigit() or parts[1] not in ("0", "1"):
        return None
    i, rel = int(parts[0]), int(parts[1])
    conf = int(parts[2]) if parts[2].isdigit() else 5
    if rel == 0:
        return i, {"relevant": 0, "confidence": min(conf, 9), "category": "", "urgency": 0, "locations": [],
                   "hazard": None, "cc": None}
    if len(parts) >= 7 and parts[3].upper() in HAZARD_CODES and parts[4].upper() in CATEGORY_CODES:  # v4 line
        cc = parts[6].upper() if re.fullmatch(r"[A-Za-z]{2}", parts[6]) else None
        loc_field = parts[7] if len(parts) > 7 else ""
        return i, {"relevant": 1, "confidence": min(conf, 9), "category": parts[4].upper(),
                   "urgency": int(parts[5]) if parts[5] in ("0", "1", "2", "3") else 1, "hazard": parts[3].upper(),
                   "cc": cc, "locations": [x.strip() for x in loc_field.split(";") if x.strip() and x.strip() != "-"]}
    rest = [p for p in parts[3:] if p != "-"]
    cat = next((p for p in rest if p.upper() in CATEGORY_CODES), "OTHER").upper()
    urg = next((int(p) for p in rest if p in ("0", "1", "2", "3")), 1)
    loc_field = next((p for p in reversed(parts[3:]) if p not in ("-", "") and p.upper() not in CATEGORY_CODES
                      and p not in ("0", "1", "2", "3")), "")
    locs = [s.strip() for s in loc_field.split(";") if s.strip() and s.strip() != "-"]
    return i, {"relevant": 1, "confidence": min(conf, 9), "category": cat, "urgency": urg, "locations": locs,
               "hazard": None, "cc": None}


def parse_labels(text: str, n: int) -> dict[int, dict]:
    out = {}
    for line in text.splitlines():
        r = _parse_line(line)
        if r and 0 <= r[0] < n and r[0] not in out:
            out[r[0]] = r[1]
    return out


def label_batch(client: Gemini, tweets: list[str], tag: str, event_hint: str | None = None,
                retry_missing: bool = True, v4: bool = False) -> dict[int, dict]:
    """Label one batch. Missing ids get one follow-up call (only if the gap is large enough to matter)."""
    prompt = (label_prompt_v4 if v4 else label_prompt)(tweets, event_hint)
    res = parse_labels(client.generate(prompt, tag=tag), len(tweets))
    missing = [i for i in range(len(tweets)) if i not in res]
    if retry_missing and missing and not client.dry_run and len(missing) >= 3:
        sub = label_batch(client, [tweets[i] for i in missing], tag + ":retry", event_hint, retry_missing=False, v4=v4)
        res.update({missing[j]: v for j, v in sub.items()})
    return res
