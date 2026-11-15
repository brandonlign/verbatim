"""Run Whisper systems over a split of ASCEND and save every raw output (resumable).

usage: run_asr.py <parquet> <out_jsonl> [--only-mixed] [--min-dur 1.0]
"""
import argparse, io, json, os
import pandas as pd, soundfile as sf, mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
CS_PROMPT = "我们今天讨论一下这个project，然后我觉得deadline有点tight。OK, so 你那个research方向是什么？"
SYSTEMS = {
    "auto": dict(),
    "force_zh": dict(language="zh"),
    "prompt_zh": dict(language="zh", initial_prompt=CS_PROMPT),
}

ap = argparse.ArgumentParser()
ap.add_argument("parquet"); ap.add_argument("out")
ap.add_argument("--only-mixed", action="store_true")
ap.add_argument("--min-dur", type=float, default=1.0)
a = ap.parse_args()

df = pd.read_parquet(a.parquet)
if a.only_mixed:
    df = df[df.language == "mixed"]
df = df[df.duration >= a.min_dur]

done = set()
if os.path.exists(a.out):
    with open(a.out) as f:
        done = {json.loads(l)["id"] for l in f}

with open(a.out, "a") as f:
    for i, (_, r) in enumerate(df.iterrows()):
        if r.id in done:
            continue
        audio, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        rec = dict(id=r.id, ref=r.transcription, dur=float(r.duration), lang=r.language,
                   speaker=int(r.original_speaker_id), session=int(r.session_id))
        for name, kw in SYSTEMS.items():
            res = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, condition_on_previous_text=False, **kw)
            rec[name] = res["text"].strip()
            rec[name + "_lang"] = res.get("language")
            segs = res.get("segments") or []
            rec[name + "_avg_logprob"] = (sum(s["avg_logprob"] for s in segs) / len(segs)) if segs else None
            rec[name + "_no_speech"] = (max(s["no_speech_prob"] for s in segs)) if segs else None
            rec[name + "_compression"] = (max(s["compression_ratio"] for s in segs)) if segs else None
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(i, rec["id"], flush=True)
