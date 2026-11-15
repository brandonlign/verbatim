"""Pilot: how does Whisper handle Mandarin-English code-switched speech (ASCEND test, 'mixed' utterances)?"""
import io, json, re, sys, random
import numpy as np, pandas as pd, soundfile as sf, jiwer, mlx_whisper
from opencc import OpenCC

DATA = sys.argv[1]
OUT = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 80
MODEL = "mlx-community/whisper-large-v3-turbo"
t2s = OpenCC("t2s")

def tokens(text):
    text = t2s.convert(text.lower())
    text = re.sub(r"[^\w\s']", " ", text)
    out = []
    for chunk in text.split():
        # split CJK characters individually, keep latin words whole
        for piece in re.findall(r"[一-鿿]|[a-z0-9']+", chunk):
            out.append(piece)
    return out

def mer(ref, hyp):
    r, h = tokens(ref), tokens(hyp)
    if not r:
        return None
    return jiwer.wer(" ".join(r), " ".join(h) if h else "<empty>")

def en_words(t):
    return [w for w in tokens(t) if re.match(r"[a-z]", w) and w not in {"呃", "嗯"}]

def en_recall(ref, hyp):
    r = en_words(ref)
    if not r:
        return None
    h = set(en_words(hyp))
    return sum(w in h for w in r) / len(r)

df = pd.read_parquet(DATA)
mixed = df[(df.language == "mixed") & (df.duration >= 2.0)]
random.seed(0)
idx = random.sample(list(mixed.index), min(N, len(mixed)))
sub = mixed.loc[idx]

conds = {
    "auto": dict(),
    "force_zh": dict(language="zh"),
    "force_en": dict(language="en"),
}
rows = []
for i, (_, r) in enumerate(sub.iterrows()):
    audio, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    assert sr == 16000, sr
    row = dict(id=r.id, ref=r.transcription, dur=r.duration)
    for name, kw in conds.items():
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, condition_on_previous_text=False, **kw)
        hyp = res["text"].strip()
        row[name] = hyp
        row[name + "_lang"] = res.get("language")
        row[name + "_mer"] = mer(r.transcription, hyp)
        row[name + "_enrec"] = en_recall(r.transcription, hyp)
    rows.append(row)
    print(i, json.dumps({k: row[k] for k in ["ref", "auto", "force_zh"]}, ensure_ascii=False), flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("\n=== SUMMARY (n=%d mixed utterances, %s) ===" % (len(out), MODEL))
for name in conds:
    m = out[name + "_mer"].dropna()
    e = out[name + "_enrec"].dropna()
    dropped = (out[name + "_enrec"] == 0).sum()
    print(f"{name:9s} MER mean={m.mean():.3f} median={m.median():.3f} | English-word recall={e.mean():.3f} | utterances with ALL English dropped={dropped}/{e.notna().sum()}")
