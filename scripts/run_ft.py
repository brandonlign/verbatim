"""Decode a parquet (ASCEND or ServiceNow) with the fine-tuned model; merge into an existing run file as key 'ft'.

usage: run_ft.py <model_dir> <parquet> <in.jsonl> <out.jsonl> [--lang zh]
"""
import argparse, io, json
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("parquet"); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--lang", default="zh"); ap.add_argument("--bs", type=int, default=8)
a = ap.parse_args()

dev = torch.device("mps")
proc = WhisperProcessor.from_pretrained(a.model)
model = WhisperForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()

df = pd.read_parquet(a.parquet)
idcol = "id" if "id" in df.columns else "ID"
audio_by_id = dict(zip(df[idcol], df["audio"]))
recs = [json.loads(l) for l in open(a.inp)]

def load(b):
    x, sr = sf.read(io.BytesIO(b["bytes"]), dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    if sr != 16000:
        import soxr
        x = soxr.resample(x, sr, 16000).astype(np.float32)
    return x

PREFIX = proc.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", f"<|{a.lang}|>", "<|transcribe|>", "<|notimestamps|>"])
EOT = proc.tokenizer.eos_token_id

@torch.no_grad()
def greedy(feats, max_new=200):
    """Plain greedy decoding with a KV cache (HF generate() crashes on this transformers build)."""
    enc = model.model.encoder(feats).last_hidden_state
    B = feats.shape[0]
    ids = torch.tensor([PREFIX] * B, device=dev)
    out = [[] for _ in range(B)]; lp = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=dev)
    past, step_in = None, ids
    for _ in range(max_new):
        o = model.model.decoder(input_ids=step_in, encoder_hidden_states=enc, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logp = torch.log_softmax(model.proj_out(o.last_hidden_state[:, -1]).float(), -1)
        nxt = logp.argmax(-1)
        for i in range(B):
            if not done[i]:
                lp[i].append(logp[i, nxt[i]].item())
                if nxt[i].item() == EOT: done[i] = True
                else: out[i].append(nxt[i].item())
        if done.all(): break
        step_in = nxt[:, None]
    texts = proc.tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts, [sum(x) / len(x) if x else None for x in lp]

with open(a.out, "w") as f:
    for s in range(0, len(recs), a.bs):
        chunk = recs[s:s + a.bs]
        feats = proc.feature_extractor([load(audio_by_id[r["id"]]) for r in chunk], sampling_rate=16000,
                                       return_tensors="pt").input_features.to(dev).half()
        texts, lps = greedy(feats)
        for r, t, lp in zip(chunk, texts, lps):
            r["ft"] = t.strip()
            r["ft_avg_logprob"] = lp
            r["ft_compression"] = None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); print(s + len(chunk), flush=True)
