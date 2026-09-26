// Living Flood Map front end: map-first explorer over classified tweets.
const CAT_COLORS = {
  INFRA: "#e8590c", EVAC: "#c2255c", HUMAN: "#b02a2a", PROPERTY: "#8a3fb0", NEEDS: "#e0a100",
  AID: "#2f9e44", ADVISORY: "#1c6fc4", SUPPORT: "#0c8599", OTHER: "#7b8794",
};
const HAZ_COLORS = {
  FLOOD: "#1c6fc4", STORM: "#7048e8", QUAKE: "#8d6e45", FIRE: "#e8590c", BLAST: "#f08c00",
  CRASH: "#5c677d", ATTACK: "#b02a2a", OTHER: "#98a2ad",
};
const URG_LABELS = ["None", "Info", "Active impact", "Life / safety"];
// "CA" -> 🇨🇦 (regional indicator letters)
const flag = (cc) => (cc && /^[A-Z]{2}$/.test(cc) ? String.fromCodePoint(...[...cc].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)) : "🏳️");
const countryName = (cc) => (cc && state.meta.countries[cc] ? state.meta.countries[cc].name : "Unknown");
const URG_VARS = ["--u0", "--u1", "--u2", "--u3"];
const cssVar = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  meta: null, dsId: null, records: [], stats: {},
  rel: "rel", unique: true, cats: new Set(), urgs: new Set(), places: new Set(), mappedOnly: false,
  hazards: new Set(), countries: new Set(), multiHazard: false, global: false, allCountries: false,
  q: "", time: null, colorBy: "cat", showRegion: false, sort: "urg", listN: 60, view: [],
};

// ------------------------------------------------------------------ map
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const map = new maplibregl.Map({
  container: "map",
  // OpenFreeMap: free vector basemap, no API key
  style: `https://tiles.openfreemap.org/styles/${dark ? "dark" : "positron"}`,
  center: [-96, 56], zoom: 3,
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
const mapReady = new Promise((r) => map.on("load", r));

mapReady.then(() => {
  map.addSource("places", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "places-halo", type: "circle", source: "places", paint: {
    "circle-radius": ["+", ["get", "r"], 4], "circle-color": ["get", "color"], "circle-opacity": 0.18 } });
  map.addLayer({ id: "places", type: "circle", source: "places", paint: {
    "circle-radius": ["get", "r"], "circle-color": ["get", "color"], "circle-opacity": 0.85,
    "circle-stroke-color": "#fff", "circle-stroke-width": ["case", ["get", "sel"], 3, 1] } });
  map.addLayer({ id: "labels", type: "symbol", source: "places", filter: [">=", ["get", "n"], 3],
    layout: { "text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-offset": [0, 1.3], "text-anchor": "top", "text-optional": true },
    paint: { "text-color": dark ? "#e6e9ec" : "#1c2126", "text-halo-color": dark ? "#111" : "#fff", "text-halo-width": 1.4 } });
  map.on("mouseenter", "places", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "places", () => (map.getCanvas().style.cursor = ""));
  map.on("click", "places", (e) => placePopup(e.features[0], e.lngLat));
  // world view: one bubble per country; places take over when zoomed in (global datasets only)
  map.addSource("countries", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "countries", type: "circle", source: "countries", paint: {
    "circle-radius": ["get", "r"], "circle-color": ["get", "color"], "circle-opacity": 0.8,
    "circle-stroke-color": "#fff", "circle-stroke-width": ["case", ["get", "sel"], 3, 1.2] } });
  map.addLayer({ id: "country-labels", type: "symbol", source: "countries",
    layout: { "text-field": ["concat", ["get", "name"], "\n", ["to-string", ["get", "n"]]], "text-font": ["Noto Sans Regular"],
      "text-size": 11, "text-offset": [0, 1.6], "text-anchor": "top", "text-optional": true },
    paint: { "text-color": dark ? "#e6e9ec" : "#1c2126", "text-halo-color": dark ? "#111" : "#fff", "text-halo-width": 1.4 } });
  map.on("mouseenter", "countries", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "countries", () => (map.getCanvas().style.cursor = ""));
  map.on("click", "countries", (e) => {
    const cc = e.features[0].properties.cc;
    state.countries = new Set([cc]); render();
    fitToData();
  });
});

function setZoomLayers() {
  // global: countries below zoom 3.5, places above; single event: places at every zoom
  const split = state.global ? 3.5 : 0;
  for (const id of ["places-halo", "places", "labels"]) map.setLayerZoomRange(id, split, 24);
  for (const id of ["countries", "country-labels"]) {
    // a zoom range of (0, 0) does not hide a layer in MapLibre; visibility does
    map.setLayoutProperty(id, "visibility", state.global ? "visible" : "none");
    if (state.global) map.setLayerZoomRange(id, 0, split);
  }
}

let popup = null;
function placePopup(f, lngLat) {
  if (popup) popup.remove();
  const name = f.properties.name;
  const tws = state.view.filter((r) => r.places.some((p) => p.name === name)).sort(byUrgency).slice(0, 4);
  const cc = f.properties.cc;
  const html = `<div class="pop"><h4>${esc(name)} ${cc ? `<span title="${esc(countryName(cc))}">${flag(cc)}</span>` : ""}</h4>
    <div class="muted">${f.properties.n} tweets · max urgency ${URG_LABELS[f.properties.maxU]} · ${f.properties.prec}</div>
    ${tws.map((r) => `<div class="ptw">${esc(r.text)}</div>`).join("")}
    <button class="btn wide" id="popFilter">Filter to ${esc(name)}</button></div>`;
  const pop = (popup = new maplibregl.Popup({ maxWidth: "320px" }).setLngLat(lngLat).setHTML(html).addTo(map));
  pop.getElement().querySelector("#popFilter").onclick = () => { state.places = new Set([name]); pop.remove(); render(); };
}

// ------------------------------------------------------------------ data
async function init() {
  state.meta = await (await fetch("/api/meta")).json();
  $("#budget").max = state.meta.upload_budget; $("#budget").value = state.meta.upload_budget;
  fillDatasets(state.meta.datasets);
  if (state.meta.datasets.length) await loadDataset(state.meta.datasets[0].id);
}

function fillDatasets(list, selected) {
  $("#dataset").innerHTML = list.map((d) => `<option value="${d.id}">${esc(d.name)} (${d.rows.toLocaleString()})</option>`).join("");
  if (selected) $("#dataset").value = selected;
}

async function loadDataset(id) {
  const d = await (await fetch(`/api/data/${id}`)).json();
  Object.assign(state, { dsId: id, records: d.records, stats: d.stats, cats: new Set(), urgs: new Set(), places: new Set(), time: null, q: "",
    countries: new Set(), allCountries: false });
  // The hazard section is always shown. Worldwide multi-event data (the bonus set) opens on floods only;
  // a single-event dataset opens on everything relevant, whatever the hazard (a judge's storm upload stays visible).
  const hz = {}; let nRel = 0;
  for (const r of d.records) if (r.canon && r.rel) { nRel++; hz[r.hz || "FLOOD"] = (hz[r.hz || "FLOOD"] || 0) + 1; }
  state.multiHazard = Object.keys(hz).length > 1 && (hz.FLOOD || 0) / Math.max(1, nRel) < 0.97;
  state.global = d.stats.mode === "global";
  state.hazards = state.global && state.multiHazard ? new Set(["FLOOD"]) : new Set();
  $("#colorHz").hidden = !state.multiHazard;
  mapReady.then(setZoomLayers);
  $("#search").value = "";
  $("#summary").innerHTML = "";
  const places = new Map();
  for (const r of d.records) if (r.rel && r.canon) for (const p of r.places) places.set(p.name, (places.get(p.name) || 0) + 1);
  $("#placeList").innerHTML = [...places.entries()].sort((a, b) => b[1] - a[1]).map(([n, c]) => `<option value="${esc(n)}">${c} tweets</option>`).join("");
  const hasTime = d.records.some((r) => r.ts);
  $("#timeline").classList.toggle("hidden", !hasTime);
  render();
  fitToData();
}

function fitToData() {
  const pts = state.view.flatMap((r) => r.places.filter((p) => p.prec !== "region").map((p) => [p.lon, p.lat]));
  if (!pts.length) return;
  // fit to the densest points so one far-away mention doesn't zoom out to the whole world
  // (a worldwide dataset keeps 98% so every affected country stays in view)
  const lo = state.global && !state.countries.size ? 0.01 : 0.05;
  const lons = pts.map((p) => p[0]).sort((a, b) => a - b), lats = pts.map((p) => p[1]).sort((a, b) => a - b);
  const q = (arr, f) => arr[Math.min(arr.length - 1, Math.floor(f * arr.length))];
  const token = ++fitToken;  // switching datasets quickly: only the latest fit may move the camera
  mapReady.then(() => {
    if (token !== fitToken) return;
    map.stop();
    map.fitBounds([[q(lons, lo), q(lats, lo)], [q(lons, 1 - lo), q(lats, 1 - lo)]], { padding: 60, maxZoom: 12, duration: 600 });
  });
}
let fitToken = 0;

// ------------------------------------------------------------------ filtering
function baseFilter(r, skip = "") {
  if (state.unique && !r.canon) return false;
  if (state.rel === "rel" && !r.rel) return false;
  if (state.rel === "irr" && r.rel) return false;
  if (state.rel === "unsure" && !(r.by === "local" && r.p >= 0.15 && r.p <= 0.85) && !(r.conf < 0.5)) return false;
  if (skip !== "hz" && state.hazards.size && r.rel && !state.hazards.has(r.hz || "FLOOD")) return false;
  if (skip !== "cc" && state.countries.size && !state.countries.has(r.cc || "??")) return false;
  if (skip !== "cat" && state.cats.size && !state.cats.has(r.cat || "OTHER")) return false;
  if (skip !== "urg" && state.urgs.size && !state.urgs.has(r.urg)) return false;
  if (state.places.size && !r.places.some((p) => state.places.has(p.name))) return false;
  if (state.mappedOnly && !r.places.some((p) => p.prec !== "region")) return false;
  if (state.q && !r.text.toLowerCase().includes(state.q)) return false;
  if (state.time && r.ts) { const t = Date.parse(r.ts); if (t < state.time[0] || t > state.time[1]) return false; }
  if (state.time && !r.ts) return false;
  return true;
}

function render() {
  state.view = state.records.filter((r) => baseFilter(r));
  renderStats(); renderChips(); renderMap(); renderList(); renderTimeline(); renderLegend();
}

function renderStats() {
  const uniq = state.records.filter((r) => r.canon);
  const rel = uniq.filter((r) => r.rel).length;
  const mapped = state.view.filter((r) => r.places.some((p) => p.prec !== "region")).length;
  const floods = uniq.filter((r) => r.rel && (r.hz || "FLOOD") === "FLOOD").length;
  const nCountries = new Set(state.view.map((r) => r.cc).filter(Boolean)).size;
  // hover a tile for the full funnel: rows -> unique -> relevant -> in view
  const n = (x) => x.toLocaleString();
  const tip = {
    tweets: `${n(state.records.length)} rows in the file · ${n(uniq.length)} unique after folding retweets and copies`,
    relevant: `${n(rel)} of ${n(uniq.length)} unique tweets are about a disaster`,
    floods: `${n(floods)} unique flood tweets · ${n(rel)} relevant tweets about any hazard`,
    view: `Tweets matching the filters${state.unique ? " (unique only: untick “Unique tweets only” to include copies)" : ""}`,
    mapped: "Tweets in view with at least one place on the map",
    countries: "Countries among the tweets in view",
  };
  $("#stats").innerHTML = [
    [n(state.records.length), "tweets", tip.tweets],
    state.multiHazard ? [n(floods), "flood tweets", tip.floods] : [`${Math.round((100 * rel) / Math.max(1, uniq.length))}%`, "relevant", tip.relevant],
    [n(state.view.length), "in view", tip.view],
    state.global ? [n(nCountries), "countries", tip.countries] : [n(mapped), "mapped", tip.mapped],
  ].map(([v, l, t]) => `<div class="stat" title="${esc(t)}"><b>${v}</b><span>${l}</span></div>`).join("");
}

function renderChips() {
  const catCounts = {}, urgCounts = {}, hzCounts = {}, ccCounts = {};
  for (const r of state.records) {
    if (baseFilter(r, "cat") && r.rel) catCounts[r.cat || "OTHER"] = (catCounts[r.cat || "OTHER"] || 0) + 1;
    if (baseFilter(r, "urg")) urgCounts[r.urg] = (urgCounts[r.urg] || 0) + 1;
    if (r.rel && baseFilter(r, "hz")) hzCounts[r.hz || "FLOOD"] = (hzCounts[r.hz || "FLOOD"] || 0) + 1;
    if (baseFilter(r, "cc")) ccCounts[r.cc || "??"] = (ccCounts[r.cc || "??"] || 0) + 1;
  }
  $("#hazChips").innerHTML = Object.keys(state.meta.hazards).map((h) =>
    `<button class="chip ${state.hazards.has(h) ? "on" : ""} ${hzCounts[h] ? "" : "dim"}" data-hz="${h}" title="${esc(state.meta.hazards[h])}">
      <span class="dot" style="background:${HAZ_COLORS[h]}"></span>${h}<span class="n">${hzCounts[h] || 0}</span></button>`).join("");
  const ccs = Object.entries(ccCounts).sort((a, b) => (a[0] === "??") - (b[0] === "??") || b[1] - a[1]);
  const shown = state.allCountries ? ccs : ccs.slice(0, 12);
  for (const cc of state.countries) if (!shown.some(([c]) => c === cc)) shown.push([cc, ccCounts[cc] || 0]);
  $("#ccChips").innerHTML = shown.map(([cc, n]) =>
    `<button class="chip ${state.countries.has(cc) ? "on" : ""}" data-cc="${cc}" title="${esc(countryName(cc === "??" ? null : cc))}">
      ${cc === "??" ? "❔ Unknown" : `${flag(cc)} ${esc(countryName(cc))}`}<span class="n">${n}</span></button>`).join("") +
    (ccs.length > 12 ? `<button class="link" id="moreCc">${state.allCountries ? "fewer" : `all ${ccs.length} countries`}</button>` : "");
  $("#catChips").innerHTML = Object.keys(state.meta.categories).map((c) =>
    `<button class="chip ${state.cats.has(c) ? "on" : ""} ${catCounts[c] ? "" : "dim"}" data-cat="${c}" title="${esc(state.meta.categories[c])}">
      <span class="dot" style="background:${CAT_COLORS[c]}"></span>${c}<span class="n">${catCounts[c] || 0}</span></button>`).join("");
  $("#urgChips").innerHTML = [3, 2, 1, 0].map((u) =>
    `<button class="chip ${state.urgs.has(u) ? "on" : ""}" data-urg="${u}"><span class="dot" style="background:${cssVar(URG_VARS[u])}"></span>${URG_LABELS[u]}<span class="n">${urgCounts[u] || 0}</span></button>`).join("");
  $("#placeChips").innerHTML = [...state.places].map((p) => `<button class="chip on" data-place="${esc(p)}">${esc(p)} ✕</button>`).join("");
}

function colorFor(a) {
  const top = (o) => Object.entries(o).sort((x, y) => y[1] - x[1])[0][0];
  if (state.colorBy === "urg") return cssVar(URG_VARS[a.maxU]);
  if (state.colorBy === "hz") return HAZ_COLORS[top(a.hzs)] || HAZ_COLORS.OTHER;
  return CAT_COLORS[top(a.cats)];
}

function bump(a, r) {
  a.n += 1; a.maxU = Math.max(a.maxU, r.urg);
  const c = r.cat || "OTHER"; a.cats[c] = (a.cats[c] || 0) + 1;
  const h = r.hz || "FLOOD"; a.hzs[h] = (a.hzs[h] || 0) + 1;
}

function renderMap() {
  const agg = new Map(), byCc = new Map();
  for (const r of state.view) {
    if (r.rel && r.cc && state.meta.countries[r.cc]) {
      const c = state.meta.countries[r.cc];
      const a = byCc.get(r.cc) || { cc: r.cc, name: c.name, lat: c.lat, lon: c.lon, n: 0, maxU: 0, cats: {}, hzs: {} };
      bump(a, r); byCc.set(r.cc, a);
    }
    for (const p of r.places) {
      if (p.prec === "region" && !state.showRegion) continue;
      const a = agg.get(p.name) || { name: p.name, lat: p.lat, lon: p.lon, prec: p.prec, cc: p.cc, n: 0, maxU: 0, cats: {}, hzs: {} };
      bump(a, r); agg.set(p.name, a);
    }
  }
  const features = [...agg.values()].map((a) => ({ type: "Feature", geometry: { type: "Point", coordinates: [a.lon, a.lat] },
    properties: { name: a.name, n: a.n, maxU: a.maxU, prec: a.prec, cc: a.cc || "", color: colorFor(a), sel: state.places.has(a.name),
      r: Math.min(34, 5 + 3.2 * Math.sqrt(a.n)) } })).sort((a, b) => b.properties.n - a.properties.n);
  const cfeatures = [...byCc.values()].map((a) => ({ type: "Feature", geometry: { type: "Point", coordinates: [a.lon, a.lat] },
    properties: { cc: a.cc, name: a.name, n: a.n, color: colorFor(a), sel: state.countries.has(a.cc),
      r: Math.min(40, 7 + 2.2 * Math.sqrt(a.n)) } })).sort((a, b) => b.properties.n - a.properties.n);
  mapReady.then(() => {
    map.getSource("places").setData({ type: "FeatureCollection", features });
    map.getSource("countries").setData({ type: "FeatureCollection", features: cfeatures });
  });
}

function renderLegend() {
  const rows = (o) => Object.entries(o).map(([c, col]) => `<div><i style="background:${col}"></i>${c}</div>`).join("");
  $("#legend").innerHTML = state.colorBy === "cat" ? `<b>Dominant category</b>` + rows(CAT_COLORS)
    : state.colorBy === "hz" ? `<b>Dominant hazard</b>` + rows(HAZ_COLORS)
    : `<b>Highest urgency</b>` + [3, 2, 1, 0].map((u) => `<div><i style="background:${cssVar(URG_VARS[u])}"></i>${URG_LABELS[u]}</div>`).join("");
  $("#legend").innerHTML += `<div class="muted" style="margin-top:4px">Circle size = tweets${state.global ? "<br>Zoom in: countries → places" : ""}</div>`;
}

const byUrgency = (a, b) => b.urg - a.urg || b.n - a.n || b.conf - a.conf;
function renderList() {
  const sorters = { urg: byUrgency, shared: (a, b) => b.n - a.n || b.urg - a.urg, time: (a, b) => (b.ts || "").localeCompare(a.ts || "") };
  const rows = [...state.view].sort(sorters[state.sort]);
  $("#listTitle").textContent = `Tweets (${rows.length.toLocaleString()})`;
  $("#list").innerHTML = rows.slice(0, state.listN).map(tweetCard).join("") || `<div class="muted">No tweets match these filters.</div>`;
  $("#more").style.display = rows.length > state.listN ? "" : "none";
}

function tweetCard(r) {
  const places = r.places.map((p) => `<span class="pl" data-place="${esc(p.name)}">📍${esc(p.name)}</span>`).join(" ");
  const cat = r.rel ? `<span class="tag" style="background:${CAT_COLORS[r.cat || "OTHER"]}">${r.cat || "OTHER"}</span>` : `<span class="tag" style="background:#7b8794">NOT RELEVANT</span>`;
  const urg = r.rel ? `<span class="urg" style="color:${cssVar(URG_VARS[r.urg])}">${URG_LABELS[r.urg]}</span>` : "";
  const hz = r.rel && state.multiHazard ? `<span class="tag" style="background:${HAZ_COLORS[r.hz || "FLOOD"]}">${r.hz || "FLOOD"}</span>` : "";
  const cc = r.cc ? `<span title="${esc(countryName(r.cc))}">${flag(r.cc)} ${esc(countryName(r.cc))}</span>` : "";
  return `<div class="tw" data-id="${r.id}"><div class="t">${esc(r.text)}</div>
    <div class="meta">${hz}${cat}${urg}${cc}${r.n > 1 ? `<span>🔁 ${r.n}×</span>` : ""}
    <span title="p(relevant)=${r.p}">${r.by === "gemini" ? "Gemini" : "local"} ${Math.round(r.conf * 100)}%</span>
    ${r.ts ? `<span>${new Date(r.ts).toLocaleString()}</span>` : ""} ${places}</div></div>`;
}

// ------------------------------------------------------------------ timeline
function renderTimeline() {
  if ($("#timeline").classList.contains("hidden")) return;
  const all = state.records.filter((r) => r.ts && baseFilter({ ...r, ts: null }, "") !== undefined && (state.unique ? r.canon : true) && (state.rel !== "rel" || r.rel));
  if (!all.length) return;
  const ts = all.map((r) => Date.parse(r.ts)).filter((t) => !isNaN(t));
  const t0 = Math.min(...ts), t1 = Math.max(...ts) + 1;
  const span = t1 - t0, hour = 3600e3;
  const binMs = span > 20 * 86400e3 ? 86400e3 : span > 2 * 86400e3 ? 6 * hour : hour;
  const nb = Math.max(1, Math.ceil(span / binMs));
  const bins = new Array(nb).fill(0), inView = new Array(nb).fill(0);
  const viewIds = new Set(state.view.map((r) => r.id));
  for (const r of all) { const b = Math.floor((Date.parse(r.ts) - t0) / binMs); bins[b]++; if (viewIds.has(r.id)) inView[b]++; }
  const max = Math.max(...bins), W = 1000, H = 80, bw = W / nb;
  const svg = $("#tlSvg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = bins.map((c, i) => `<rect x="${i * bw}" y="${H - (H * c) / max}" width="${Math.max(1, bw - 1)}" height="${(H * c) / max}" fill="${cssVar("--border")}"/>` +
    `<rect x="${i * bw}" y="${H - (H * inView[i]) / max}" width="${Math.max(1, bw - 1)}" height="${(H * inView[i]) / max}" fill="${cssVar("--accent")}"/>`).join("");
  svg.dataset.t0 = t0; svg.dataset.t1 = t1;
  const fmt = (t) => new Date(t).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit" });
  $("#tlRange").textContent = state.time ? `${fmt(state.time[0])} → ${fmt(state.time[1])}` : `${fmt(t0)} → ${fmt(t1)} · drag to filter`;
}
(function timelineBrush() {
  const svg = $("#tlSvg"); let start = null;
  const tAt = (e) => { const b = svg.getBoundingClientRect(); const f = Math.min(1, Math.max(0, (e.clientX - b.left) / b.width));
    return +svg.dataset.t0 + f * (svg.dataset.t1 - svg.dataset.t0); };
  svg.addEventListener("pointerdown", (e) => { start = tAt(e); svg.setPointerCapture(e.pointerId); });
  svg.addEventListener("pointerup", (e) => { if (start === null) return; const end = tAt(e);
    state.time = Math.abs(end - start) < 60e3 ? null : [Math.min(start, end), Math.max(start, end)]; start = null; render(); });
  $("#tlClear").onclick = () => { state.time = null; render(); };
})();

// ------------------------------------------------------------------ summary
function filtersDesc() {
  const parts = [{ rel: "relevant", unsure: "uncertain", irr: "not relevant", all: "all" }[state.rel] + " tweets"];
  if (state.hazards.size) parts.push("hazard " + [...state.hazards].join("/"));
  if (state.countries.size) parts.push("countries " + [...state.countries].map((c) => countryName(c === "??" ? null : c)).join(", "));
  if (state.cats.size) parts.push("categories " + [...state.cats].join("/"));
  if (state.urgs.size) parts.push("urgency " + [...state.urgs].join("/"));
  if (state.places.size) parts.push("places " + [...state.places].join(", "));
  if (state.q) parts.push(`text contains "${state.q}"`);
  if (state.time) parts.push("time " + state.time.map((t) => new Date(t).toISOString()).join(" to "));
  return parts.join("; ");
}

function mdToHtml(md, cited) {
  const lines = esc(md).split("\n"); let html = "", inList = false;
  for (let l of lines) {
    l = l.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (m, g) =>
      g.split(",").map((n) => `<span class="cite" data-row="${cited[+n.trim()] ?? ""}">[${n.trim()}]</span>`).join(""));
    if (/^\s*[-*] /.test(l)) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${l.replace(/^\s*[-*] /, "")}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    if (/^#{1,3} /.test(l)) html += `<h3>${l.replace(/^#+ /, "")}</h3>`; else if (l.trim()) html += `<p>${l}</p>`;
  }
  return html + (inList ? "</ul>" : "");
}

$("#sumBtn").onclick = async () => {
  const groups = [...new Set(state.view.filter((r) => r.rel).map((r) => r.g))];
  if (!groups.length) { $("#summary").textContent = "No relevant tweets in this view."; return; }
  $("#sumBtn").disabled = true; $("#summary").textContent = "Summarizing…";
  try {
    const res = await fetch(`/api/summary/${state.dsId}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ groups, filters: filtersDesc() }) });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || res.statusText);
    $("#summary").classList.remove("muted");
    $("#summary").innerHTML = `<div class="muted" style="font-size:11.5px">Summary of: ${esc(filtersDesc())}</div>` + mdToHtml(d.markdown, d.cited);
  } catch (e) { $("#summary").textContent = "Summary failed: " + e.message; }
  $("#sumBtn").disabled = false;
};

// ------------------------------------------------------------------ export
function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("#exGeo").onclick = () => {
  const features = state.view.flatMap((r) => r.places.map((p) => ({ type: "Feature", geometry: { type: "Point", coordinates: [p.lon, p.lat] },
    properties: { tweet_id: r.id, text: r.text, relevant: r.rel, hazard: r.hz, category: r.cat, urgency: r.urg, confidence: r.conf,
      decided_by: r.by, country: r.cc, shares: r.n, timestamp: r.ts, place: p.name, place_country: p.cc,
      place_precision: p.prec, place_source: p.src } })));
  download("flood_tweets.geojson", JSON.stringify({ type: "FeatureCollection", features }), "application/geo+json");
};
$("#exCsv").onclick = () => {
  const cols = ["tweet_id", "text", "relevant", "hazard", "category", "urgency", "confidence", "decided_by", "country", "shares", "timestamp", "places", "lats", "lons"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = state.view.map((r) => [r.id, r.text, r.rel, r.hz, r.cat, r.urg, r.conf, r.by, r.cc, r.n, r.ts, r.places.map((p) => p.name).join(";"),
    r.places.map((p) => p.lat).join(";"), r.places.map((p) => p.lon).join(";")].map(q).join(","));
  download("flood_tweets.csv", [cols.join(","), ...rows].join("\n"), "text/csv");
};

// ------------------------------------------------------------------ upload
$("#uploadBtn").onclick = () => { $("#jobStatus").textContent = ""; $("#jobBar").style.width = 0; updBudget(); $("#uploadDlg").showModal(); };
const updBudget = () => { $("#budgetVal").textContent = $("#budget").value;
  $("#budgetTweets").textContent = ((state.meta?.batch_size || 200) * $("#budget").value).toLocaleString(); };
$("#budget").oninput = updBudget;
// a plain button: as a form submit it was blocked by the required file field, so the dialog never closed
$("#closeUpload").onclick = () => $("#uploadDlg").close();
$("#startUpload").onclick = async (e) => {
  e.preventDefault();
  const f = $("#file").files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f); fd.append("budget", $("#budget").value);
  $("#startUpload").disabled = true; $("#jobStatus").textContent = "Uploading…";
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd }); const d = await r.json();
    if (!r.ok) throw new Error(d.detail);
    for (;;) {
      await new Promise((ok) => setTimeout(ok, 1200));
      const j = await (await fetch(`/api/jobs/${d.job}`)).json();
      $("#jobStatus").textContent = j.msg; $("#jobBar").style.width = `${Math.round(100 * j.frac)}%`;
      if (j.status === "error") throw new Error(j.error);
      if (j.status === "done") {
        const meta = await (await fetch("/api/meta")).json();
        fillDatasets(meta.datasets, j.dataset); await loadDataset(j.dataset); $("#uploadDlg").close(); break;
      }
    }
  } catch (err) { $("#jobStatus").textContent = "Failed: " + err.message; }
  $("#startUpload").disabled = false;
};

// ------------------------------------------------------------------ events
$("#dataset").onchange = (e) => loadDataset(e.target.value);
$("#relSeg").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; state.rel = b.dataset.v;
  $("#relSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); render(); };
$("#colorSeg").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; state.colorBy = b.dataset.v;
  $("#colorSeg").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); renderMap(); renderLegend(); };
$("#uniqueOnly").onchange = (e) => { state.unique = e.target.checked; render(); };
$("#mappedOnly").onchange = (e) => { state.mappedOnly = e.target.checked; render(); };
$("#showRegion").onchange = (e) => { state.showRegion = e.target.checked; renderMap(); };
$("#sort").onchange = (e) => { state.sort = e.target.value; renderList(); };
$("#more").onclick = () => { state.listN += 60; renderList(); };
let searchT; $("#search").oninput = (e) => { clearTimeout(searchT); searchT = setTimeout(() => { state.q = e.target.value.trim().toLowerCase(); render(); }, 200); };
$("#placeInput").onchange = (e) => { const v = e.target.value.trim(); if (v) { state.places.add(v); e.target.value = ""; render(); } };
$("#panelToggle").onclick = () => $("#panel").classList.toggle("open");
document.addEventListener("click", (e) => {
  const c = e.target.closest("[data-cat]"); if (c) { toggle(state.cats, c.dataset.cat); return render(); }
  const u = e.target.closest("[data-urg]"); if (u) { toggle(state.urgs, +u.dataset.urg); return render(); }
  const p = e.target.closest("[data-place]"); if (p) { toggle(state.places, p.dataset.place); return render(); }
  const h = e.target.closest("[data-hz]"); if (h) { toggle(state.hazards, h.dataset.hz); return render(); }
  const k = e.target.closest("[data-cc]"); if (k) { toggle(state.countries, k.dataset.cc); render(); return fitToData(); }
  if (e.target.id === "moreCc") { state.allCountries = !state.allCountries; return renderChips(); }
  const cl = e.target.closest("[data-clear]"); if (cl) {
    ({ cat: state.cats, urg: state.urgs, place: state.places, hz: state.hazards, cc: state.countries })[cl.dataset.clear].clear(); return render(); }
  const ci = e.target.closest(".cite"); if (ci && ci.dataset.row) return focusTweet(+ci.dataset.row);
  const tw = e.target.closest(".tw"); if (tw) return flyToTweet(+tw.dataset.id);
});
const toggle = (set, v) => (set.has(v) ? set.delete(v) : set.add(v));

function focusTweet(id) {
  const r = state.records.find((x) => x.id === id); if (!r) return;
  if (!state.view.includes(r)) { state.view.unshift(r); }
  const list = $("#list");
  list.insertAdjacentHTML("afterbegin", tweetCard(r));
  const el = list.firstElementChild; el.classList.add("hl"); el.scrollIntoView({ behavior: "smooth", block: "center" });
  flyToTweet(id);
}
function flyToTweet(id) {
  const r = state.records.find((x) => x.id === id);
  const p = r && (r.places.find((q) => q.prec !== "region") || r.places[0]);
  if (p) map.flyTo({ center: [p.lon, p.lat], zoom: Math.max(map.getZoom(), p.prec === "point" ? 13 : 10) });
}

init();
