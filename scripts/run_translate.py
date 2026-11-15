"""Add Whisper's explicit TRANSLATE-task output to a run file (key 'translate').
If a transcript looks like this output, the "transcript" is a translation.

usage: run_translate.py <parquet> <in.jsonl> <out.jsonl> [--lang zh]
"""
import argparse, io, json
import pandas as pd, soundfile as sf, mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
ap = argparse.ArgumentParser()
ap.add_argument("parquet"); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--lang", default="zh")
a = ap.parse_args()
df = pd.read_parquet(a.parquet).set_index("id")
recs = [json.loads(l) for l in open(a.inp)]
with open(a.out, "w") as f:
    for i, r in enumerate(recs):
        audio, sr = sf.read(io.BytesIO(df.loc[r["id"], "audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, condition_on_previous_text=False,
                                     language=a.lang, task="translate")
        segs = res.get("segments") or []
        r["translate"] = res["text"].strip()
        r["translate_avg_logprob"] = (sum(s["avg_logprob"] for s in segs) / len(segs)) if segs else None
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        print(i, flush=True)
