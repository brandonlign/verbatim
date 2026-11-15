"""Rebuild the fine-tuned model in models/ft_decoder without re-training.

Only Whisper's decoder was trained (the encoder stayed frozen), so the release holds just the 100 decoder
tensors that changed (688 MB). This script downloads them, loads the public base model, swaps them in,
and checks every tensor's checksum against the trained model.

usage: .venv-train/bin/python scripts/get_weights.py [--delta path/to/verbatim_decoder.safetensors]
"""
import argparse, hashlib, json, os, urllib.request
import torch
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-large-v3-turbo"
URL = "https://github.com/brandonlign/verbatim/releases/download/v1.0/verbatim_decoder.safetensors"
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS = os.path.join(HERE, "..", "models_sha256.json")

def tensor_sha(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", help="local copy of verbatim_decoder.safetensors (skips the download)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "models", "ft_decoder"))
    a = ap.parse_args()
    delta_path = a.delta
    if not delta_path:
        delta_path = os.path.join(HERE, "..", "models", "verbatim_decoder.safetensors")
        if not os.path.exists(delta_path):
            os.makedirs(os.path.dirname(delta_path), exist_ok=True)
            print("downloading", URL, flush=True)
            urllib.request.urlretrieve(URL, delta_path)
    delta = load_file(delta_path)
    model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
    missing = [k for k in delta if k not in model.state_dict()]
    assert not missing, f"unexpected tensors: {missing[:3]}"
    model.load_state_dict(delta, strict=False)
    want = json.load(open(CHECKSUMS))
    bad = [k for k, h in want.items() if tensor_sha(model.state_dict()[k]) != h]
    assert not bad, f"{len(bad)} tensors don't match the trained model, e.g. {bad[:3]}"
    model.save_pretrained(a.out)
    WhisperProcessor.from_pretrained(BASE).save_pretrained(a.out)
    print(f"rebuilt {a.out}: {len(delta)} fine-tuned tensors swapped in, {len(want)} checksums match", flush=True)
