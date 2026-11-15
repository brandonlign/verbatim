"""Verbatim live demo. Record mixed-language speech in the browser, compare plain Whisper with Verbatim.

usage: .venv/bin/python -I demo/server.py [--port 8780]
"""
import argparse, json, os, subprocess, sys, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, soundfile as sf, mlx_whisper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from verbatim import check_v2, pieces_from_ids, token_units  # noqa: E402

MODEL = "mlx-community/whisper-large-v3-turbo"
PROMPTS = {
    "zh": "我们今天讨论一下这个project，然后我觉得deadline有点tight。OK, so 你那个research方向是什么？",
    "es": "Hoy tengo una appointment con el doctor, y después tengo que hacer el refill de mi medicine. I'm not sure si es a las tres.",
}
PROMPT_EN = {
    "zh": "OK so 我们今天讨论一下这个project, and I think the deadline 有点 tight. 你那个 research 方向是什么?",
    "es": "OK so today I have una appointment con el doctor, and then I need to do the refill de mi medicine.",
}
TH = json.load(open(os.path.join(HERE, "..", "data", "thresholds_v2_frozen.json")))
TH3 = json.load(open(os.path.join(HERE, "..", "data", "thresholds_v3_frozen.json")))
V4_PATH = os.path.join(HERE, "..", "data", "thresholds_v4_frozen.json")
V4 = json.load(open(V4_PATH)) if os.path.exists(V4_PATH) else None  # holds the frozen word-mark cut-off
FT = None  # fine-tuned Mandarin-English model, loaded at startup if --ft is given

def decode(audio, **kw):
    r = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, condition_on_previous_text=False, **kw)
    segs = r.get("segments") or []
    return dict(text=r["text"].strip(), lang=r.get("language"),
                logprob=(sum(s["avg_logprob"] for s in segs) / len(segs)) if segs else None,
                compression=max((s["compression_ratio"] for s in segs), default=None))

def run(audio, primary):
    rec = {"dur": len(audio) / 16000}
    for key, kw in [("auto", {}), (f"prompt_{primary}", dict(language=primary, initial_prompt=PROMPTS[primary])),
                    ("prompt_en", dict(language="en", initial_prompt=PROMPT_EN[primary]))]:
        d = decode(audio, **kw)
        rec[key], rec[key + "_lang"], rec[key + "_avg_logprob"], rec[key + "_compression"] = d["text"], d["lang"], d["logprob"], d["compression"]
    out = dict(plain=rec["auto"], plain_lang=rec["auto_lang"])
    if FT is not None and primary == "zh":
        (rec["ft"],), (rec["ft_avg_logprob"],), (toks_lp,) = FT.transcribe([audio], tokens=True)
        rec["ft_pieces"] = pieces_from_ids(toks_lp, FT.proc.tokenizer)
        chosen, flagged, reasons, key = check_v2(rec, TH3, primary=primary, cands=TH3["cands"])
        if key == "ft":  # word marks: spans of the transcript whose token log-prob is under the frozen cut-off
            cut = V4["word_logprob"] if V4 else -1.0
            text, units = token_units(rec["ft_pieces"])
            out["text"] = text; out["marks"] = [[a, b] for _, lp, a, b in units if lp < cut]
    else:
        chosen, flagged, reasons, key = check_v2(rec, TH, primary=primary)
    out.update(verbatim=out.get("text", chosen).strip() if key == "ft" else chosen, flagged=flagged, reasons=reasons, chosen_decode=key)
    if "text" in out:  # marks index into the unstripped text
        lead = len(out["text"]) - len(out["text"].lstrip()); out["marks"] = [[a - lead, b - lead] for a, b in out["marks"]]; del out["text"]
    return out

class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        path = self.path.split("?")[0]
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/examples.json": ("examples.json", "application/json")}
        if path in files:
            fn, ct = files[path]
            p = os.path.join(HERE, fn)
            if os.path.exists(p):
                return self.send(200, open(p, "rb").read(), ct)
        if path.startswith("/audio/") and path.endswith(".wav"):
            p = os.path.join(HERE, "audio", os.path.basename(path))
            if os.path.exists(p):
                return self.send(200, open(p, "rb").read(), "audio/wav")
        if path.startswith("/example/"):  # run the whole pipeline live on a stored test-set clip
            p = os.path.join(HERE, "audio", os.path.basename(path) + ".wav")
            if os.path.exists(p):
                audio, _ = sf.read(p, dtype="float32")
                return self.send(200, json.dumps(run(audio.astype(np.float32), "zh"), ensure_ascii=False), "application/json")
        self.send(404, "not found", "text/plain")
    def do_POST(self):
        if not self.path.startswith("/transcribe"):
            return self.send(404, "not found", "text/plain")
        primary = "es" if "lang=es" in self.path else "zh"
        data = self.rfile.read(int(self.headers["Content-Length"]))
        with tempfile.TemporaryDirectory() as d:
            src, wav = os.path.join(d, "in.webm"), os.path.join(d, "in.wav")
            open(src, "wb").write(data)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", wav], check=True)
            audio, _ = sf.read(wav, dtype="float32")
        out = run(audio.astype(np.float32), primary)
        self.send(200, json.dumps(out, ensure_ascii=False), "application/json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--ft", help="fine-tuned model dir (Mandarin-English); needs torch + transformers")
    args = ap.parse_args(); port = args.port
    if args.ft:
        from ft_model import FineTuned
        FT = FineTuned(args.ft)
    print(f"Verbatim demo on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
