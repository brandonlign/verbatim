"""Add one more decode to an existing run file: forced English + code-switched prompt.

usage: add_decode.py <parquet> <in.jsonl> <out.jsonl>
"""
import io, json, sys
import pandas as pd, soundfile as sf, mlx_whisper

MODEL = "mlx-community/whisper-large-v3-turbo"
CS_PROMPT_EN = "OK so 我们今天讨论一下这个project, and I think the deadline 有点 tight. 你那个 research 方向是什么?"
parquet, inp, out = sys.argv[1:4]
df = pd.read_parquet(parquet).set_index("id")
recs = [json.loads(l) for l in open(inp)]
with open(out, "w") as f:
    for i, r in enumerate(recs):
        audio, sr = sf.read(io.BytesIO(df.loc[r["id"], "audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, condition_on_previous_text=False,
                                     language="en", initial_prompt=CS_PROMPT_EN)
        segs = res.get("segments") or []
        r["prompt_en"] = res["text"].strip()
        r["prompt_en_avg_logprob"] = (sum(s["avg_logprob"] for s in segs) / len(segs)) if segs else None
        r["prompt_en_compression"] = max((s["compression_ratio"] for s in segs), default=None)
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        print(i, flush=True)
