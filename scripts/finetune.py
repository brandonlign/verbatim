"""Fine-tune Whisper large-v3-turbo's DECODER on ASCEND train (code-switched Mandarin-English).
The encoder (the part that hears) is frozen; only the part that decides what to write is trained.

usage: finetune.py <train_parquet_dir> <out_dir> [--probe] [--max-steps N] [--bs 8] [--lr 1e-5]
"""
import argparse, glob, io, math, os, random, time, json
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("train_dir"); ap.add_argument("out")
ap.add_argument("--probe", action="store_true")
ap.add_argument("--max-steps", type=int, default=1200)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--max-dur", type=float, default=12.0)
a = ap.parse_args()

dev = torch.device("mps")
MID = "openai/whisper-large-v3-turbo"
proc = WhisperProcessor.from_pretrained(MID)
model = WhisperForConditionalGeneration.from_pretrained(MID, torch_dtype=torch.float32).to(dev)
model.config.forced_decoder_ids = None
for p in model.model.encoder.parameters():
    p.requires_grad = False
model.model.encoder.half()
trainable = [p for p in model.parameters() if p.requires_grad]
print("trainable params (M):", sum(p.numel() for p in trainable) / 1e6, flush=True)

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.train_dir, "*.parquet")))])
df = df[(df.duration >= 0.8) & (df.duration <= a.max_dur)]
df = df[~df.transcription.str.contains(r"\[UNK\]")]
rows = df[["audio", "transcription"]].to_dict("records")
random.Random(0).shuffle(rows)
print("training utterances:", len(rows), flush=True)

prefix = proc.tokenizer.convert_tokens_to_ids(["<|startoftranscript|>", "<|zh|>", "<|transcribe|>", "<|notimestamps|>"])
eot = proc.tokenizer.eos_token_id

def batch(items):
    auds = []
    for it in items:
        x, sr = sf.read(io.BytesIO(it["audio"]["bytes"]), dtype="float32")
        auds.append(x if x.ndim == 1 else x.mean(1))
    feats = proc.feature_extractor(auds, sampling_rate=16000, return_tensors="pt").input_features
    labs = [prefix + proc.tokenizer.encode(" " + it["transcription"].strip(), add_special_tokens=False) + [eot] for it in items]
    L = max(map(len, labs))
    dec_in = torch.full((len(labs), L - 1), eot); tgt = torch.full((len(labs), L - 1), -100)
    for i, l in enumerate(labs):
        dec_in[i, :len(l) - 1] = torch.tensor(l[:-1]); t = torch.tensor(l[1:]); t[:3] = -100  # don't train on the prompt tokens
        tgt[i, :len(l) - 1] = t
    return feats.to(dev).half(), dec_in.to(dev), tgt.to(dev)

opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
steps = 3 if a.probe else a.max_steps
warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
model.train(); model.model.encoder.eval()
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "train_log.jsonl"), "a")
i = 0; t0 = time.time()
for step in range(steps):
    items = [rows[(i + k) % len(rows)] for k in range(a.bs)]; i += a.bs
    feats, dec_in, tgt = batch(items)
    with torch.no_grad():
        enc = model.model.encoder(feats).last_hidden_state.float()
    out = model.model.decoder(input_ids=dec_in, encoder_hidden_states=enc)
    logits = model.proj_out(out.last_hidden_state)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
    if step % 20 == 0 or a.probe:
        el = time.time() - t0
        rec = dict(step=step, loss=round(loss.item(), 4), sec_per_step=round(el / (step + 1), 2), epoch=round(i / len(rows), 3))
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
if not a.probe:
    model.model.encoder.float()
    model.save_pretrained(a.out); proc.save_pretrained(a.out)
    print("saved", a.out, flush=True)
