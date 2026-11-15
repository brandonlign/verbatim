"""Run the three decodes on a ServiceNow asr_codeswitched config (unseen test set).

usage: run_sn.py <parquet> <out.jsonl> <primary: zh|es>
"""
import io, json, os, sys
import pandas as pd, soundfile as sf, mlx_whisper, numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"
PROMPTS = {
    "zh": "我们今天讨论一下这个project，然后我觉得deadline有点tight。OK, so 你那个research方向是什么？",
    "es": "Hoy tengo una appointment con el doctor, y después tengo que hacer el refill de mi medicine. I'm not sure si es a las tres.",
}
PROMPT_EN = {
    "zh": "OK so 我们今天讨论一下这个project, and I think the deadline 有点 tight. 你那个 research 方向是什么?",
    "es": "OK so today I have una appointment con el doctor, and then I need to do the refill de mi medicine.",
}
parquet, out, primary = sys.argv[1:4]
df = pd.read_parquet(parquet)
done = set()
if os.path.exists(out):
    done = {json.loads(l)["id"] for l in open(out)}

def to16k(b):
    audio, sr = sf.read(io.BytesIO(b), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != 16000:
        import soxr
        audio = soxr.resample(audio, sr, 16000).astype(np.float32)
    return audio

def dec(audio, **kw):
    r = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, condition_on_previous_text=False, **kw)
    segs = r.get("segments") or []
    return (r["text"].strip(), r.get("language"),
            (sum(s["avg_logprob"] for s in segs) / len(segs)) if segs else None,
            max((s["compression_ratio"] for s in segs), default=None))

with open(out, "a") as f:
    for i, r in df.iterrows():
        if r.ID in done:
            continue
        audio = to16k(r.audio["bytes"])
        rec = dict(id=r.ID, ref=r.utterance, dur=len(audio) / 16000,
                   words=list(r.words), word_languages=list(r.word_languages))
        for name, kw in [("auto", {}), (f"prompt_{primary}", dict(language=primary, initial_prompt=PROMPTS[primary])),
                         ("prompt_en", dict(language="en", initial_prompt=PROMPT_EN[primary]))]:
            t, lang, lp, cr = dec(audio, **kw)
            rec[name], rec[name + "_lang"], rec[name + "_avg_logprob"], rec[name + "_compression"] = t, lang, lp, cr
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(i, flush=True)
