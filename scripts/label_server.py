"""Tiny local labeling tool. Teammates mark each transcript's failure type.

usage: label_server.py <asr_outputs.jsonl> <labels.jsonl> [--port 8770]
Open http://127.0.0.1:8770 . Keys: 1=ok 2=silent translation 3=dropped words 4=hallucination 5=mishearing, s=skip.
Systems are shown in random order and unnamed, so the labeler can't tell which system made which line.
"""
import argparse, json, os, random
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("asr"); ap.add_argument("labels"); ap.add_argument("--port", type=int, default=8770)
ap.add_argument("--systems", default="auto,force_zh,prompt_zh")
a = ap.parse_args()
SYSTEMS = a.systems.split(",")

items = []
with open(a.asr) as f:
    for line in f:
        r = json.loads(line)
        for s in SYSTEMS:
            if s in r:
                items.append(dict(id=r["id"], system=s, ref=r["ref"], hyp=r[s]))
random.Random(42).shuffle(items)

def labeled():
    done = set()
    if os.path.exists(a.labels):
        with open(a.labels) as f:
            for l in f:
                x = json.loads(l); done.add((x["id"], x["system"]))
    return done

PAGE = """<!doctype html><meta charset=utf-8><title>Label transcripts</title>
<style>body{font:18px system-ui;max-width:900px;margin:40px auto;padding:0 16px}
.box{border:1px solid #ccc;border-radius:8px;padding:16px;margin:12px 0}.k{color:#666;font-size:14px}
button{font-size:16px;margin:4px;padding:8px 12px}</style>
<div id=prog class=k></div>
<div class=box><div class=k>What was actually said (reference)</div><div id=ref></div></div>
<div class=box><div class=k>Transcript</div><div id=hyp></div></div>
<div>
<button onclick=L('ok')>1 OK / minor typo</button>
<button onclick=L('silent_translation')>2 Silent translation</button>
<button onclick=L('dropped')>3 Dropped words</button>
<button onclick=L('hallucination')>4 Hallucination</button>
<button onclick=L('mishearing')>5 Mishearing</button>
<button onclick=L('skip')>s Skip</button></div>
<p class=k>Silent translation = part of what was said appears in the OTHER language, as a translation (e.g. Chinese speech written as English).
Dropped = words missing. Mishearing = wrong words, same language. Hallucination = text that wasn't spoken at all.</p>
<script>
let cur=null;
async function next(){const r=await fetch('/next');cur=await r.json();
if(!cur.id){document.body.innerHTML='<h2>All done, thank you!</h2>';return}
ref.textContent=cur.ref;hyp.textContent=cur.hyp;prog.textContent=cur.done+' / '+cur.total+' labeled'}
async function L(lab){await fetch('/label',{method:'POST',body:JSON.stringify({id:cur.id,system:cur.system,label:lab})});next()}
document.onkeydown=e=>{const m={'1':'ok','2':'silent_translation','3':'dropped','4':'hallucination','5':'mishearing','s':'skip'};if(m[e.key])L(m[e.key])};
next();
</script>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def send(self, code, body, ctype):
        b = body.encode(); self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/":
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/next":
            done = labeled()
            for it in items:
                if (it["id"], it["system"]) not in done:
                    return self.send(200, json.dumps(dict(it, done=len(done), total=len(items)), ensure_ascii=False), "application/json")
            return self.send(200, "{}", "application/json")
        self.send(404, "not found", "text/plain")
    def do_POST(self):
        if self.path == "/label":
            x = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with open(a.labels, "a") as f:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
            return self.send(200, "{}", "application/json")
        self.send(404, "not found", "text/plain")

print(f"Labeling {len(items)} transcripts at http://127.0.0.1:{a.port}")
HTTPServer(("127.0.0.1", a.port), H).serve_forever()
