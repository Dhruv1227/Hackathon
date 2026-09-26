"""Local hand-labeling tool for the tune (100) and gold (200) sets.

Run:  .venv/bin/python -m labeling.app   ->  http://127.0.0.1:5001
Labels are saved after every tweet to data/labels/hand_labels.csv (resumable).
"""
import csv
import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from pipeline.taxonomy import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/processed/clean.csv"
OUT = ROOT / "data/labels/hand_labels.csv"
FIELDS = ["group_id", "split", "relevant", "category", "urgency", "locations"]

app = Flask(__name__)


def items():
    df = pd.read_csv(CLEAN)
    df = df[df.is_canonical & df.split.isin(["tune", "gold"])]
    # interleave so tune and gold are labeled under the same conditions
    df = df.sample(frac=1, random_state=7)
    return df[["group_id", "split", "text"]].to_dict("records")


def load_labels():
    if not OUT.exists():
        return {}
    with OUT.open() as f:
        return {int(r["group_id"]): r for r in csv.DictReader(f)}


def save_labels(labels):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(labels.values())


@app.get("/api/items")
def api_items():
    labels = load_labels()
    return jsonify([{**it, "label": labels.get(it["group_id"])} for it in items()])


@app.post("/api/label")
def api_label():
    d = request.get_json()
    labels = load_labels()
    labels[int(d["group_id"])] = {k: d.get(k, "") for k in FIELDS}
    save_labels(labels)
    return jsonify(ok=True, n=len(labels))


@app.get("/")
def index():
    return PAGE.replace("__CATS__", json.dumps(CATEGORIES))


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Flood Tweet Labeler</title>
<style>
body{font:15px system-ui;max-width:760px;margin:24px auto;padding:0 16px;color:#1b1f24}
.tweet{font-size:19px;line-height:1.45;padding:18px;border:1px solid #d0d7de;border-radius:10px;margin:14px 0;background:#f6f8fa;white-space:pre-wrap}
.row{margin:12px 0}.btn{padding:6px 11px;margin:2px;border:1px solid #d0d7de;border-radius:6px;background:#fff;cursor:pointer}
.btn.on{background:#0969da;color:#fff;border-color:#0969da}
input[type=text]{width:100%;padding:8px;font-size:15px;box-sizing:border-box}
.muted{color:#656d76;font-size:13px}kbd{background:#eee;border-radius:3px;padding:0 4px;font-size:12px}
.bar{height:6px;background:#eee;border-radius:3px}.bar>div{height:6px;background:#1a7f37;border-radius:3px}
</style></head><body>
<h2>Flood tweet labeler <span class="muted" id="prog"></span></h2>
<div class="bar"><div id="pbar"></div></div>
<div class="tweet" id="tw"></div>
<div class="row"><b>Relevant to the flood?</b>
 <button class="btn" data-f="relevant" data-v="1">Yes <kbd>y</kbd></button>
 <button class="btn" data-f="relevant" data-v="0">No <kbd>n</kbd></button></div>
<div id="relbox">
<div class="row"><b>Category</b> <span class="muted">(keys 1–9)</span><div id="cats"></div></div>
<div class="row"><b>Urgency</b> <span class="muted">(shift+0–3: 0 none · 1 info · 2 active impact · 3 life/safety now)</span><div id="urg"></div></div>
<div class="row"><b>Locations</b> <span class="muted">semicolon-separated, as named in the tweet (e.g. Inglewood;Bow River;Calgary)</span>
 <input type="text" id="loc"></div>
</div>
<div class="row"><button class="btn" id="prev">← Prev <kbd>[</kbd></button>
 <button class="btn on" id="save">Save &amp; next <kbd>Enter</kbd></button>
 <button class="btn" id="nextu">Jump to next unlabeled</button></div>
<p class="muted" id="catdesc"></p>
<script>
const CATS=__CATS__;let items=[],i=0,cur={};
const $=s=>document.querySelector(s);
Object.keys(CATS).forEach((c,k)=>{$('#cats').insertAdjacentHTML('beforeend',`<button class="btn" data-f="category" data-v="${c}" title="${CATS[c]}">${k+1} ${c}</button>`)});
[0,1,2,3].forEach(u=>$('#urg').insertAdjacentHTML('beforeend',`<button class="btn" data-f="urgency" data-v="${u}">${u}</button>`));
$('#catdesc').innerHTML=Object.entries(CATS).map(([k,v])=>`<b>${k}</b>: ${v}`).join('<br>');
document.addEventListener('click',e=>{const b=e.target.closest('[data-f]');if(b){cur[b.dataset.f]=b.dataset.v;paint()}});
function paint(){document.querySelectorAll('[data-f]').forEach(b=>b.classList.toggle('on',cur[b.dataset.f]===b.dataset.v));
 $('#relbox').style.opacity=cur.relevant==='0'?.35:1;}
function show(){const it=items[i];cur=it.label?{...it.label}:{};$('#tw').textContent=it.text;$('#loc').value=cur.locations||'';
 const done=items.filter(x=>x.label).length;$('#prog').textContent=`${i+1}/${items.length} · ${done} labeled`;
 $('#pbar').style.width=(100*done/items.length)+'%';paint();}
async function save(){const it=items[i];if(cur.relevant===undefined){alert('Pick relevant yes/no');return}
 if(cur.relevant==='1'&&(!cur.category||cur.urgency===undefined)){alert('Pick category and urgency');return}
 const rec={group_id:it.group_id,split:it.split,relevant:cur.relevant,category:cur.relevant==='1'?cur.category:'',
  urgency:cur.relevant==='1'?cur.urgency:'0',locations:$('#loc').value.trim()};
 await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)});
 it.label=rec;if(i<items.length-1)i++;show();}
$('#save').onclick=save;$('#prev').onclick=()=>{if(i>0){i--;show()}};
$('#nextu').onclick=()=>{const j=items.findIndex(x=>!x.label);if(j>=0){i=j;show()}};
document.addEventListener('keydown',e=>{if(e.target.id==='loc'){if(e.key==='Enter'){e.preventDefault();save()}return}
 const cats=Object.keys(CATS);
 if(e.key==='y'){cur.relevant='1'}else if(e.key==='n'){cur.relevant='0'}
 else if(e.shiftKey&&')!@#'.includes(e.key)){cur.urgency=String(')!@#'.indexOf(e.key))}
 else if(/^[1-9]$/.test(e.key)&&!e.shiftKey){cur.category=cats[+e.key-1]}
 else if(e.key==='Enter'){save();return}else if(e.key==='['){$('#prev').click();return}
 else if(e.key==='l'){e.preventDefault();$('#loc').focus();return}else return;paint();});
fetch('/api/items').then(r=>r.json()).then(d=>{items=d;const j=items.findIndex(x=>!x.label);i=j<0?0:j;show()});
</script></body></html>"""

if __name__ == "__main__":
    app.run(port=5001, debug=False)
