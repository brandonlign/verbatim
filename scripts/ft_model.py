"""Load the fine-tuned Whisper (decoder trained on ASCEND) and transcribe 16 kHz audio with greedy decoding."""
import numpy as np, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

class FineTuned:
    def __init__(self, model_dir, lang="zh"):
        self.dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.proc = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16).to(self.dev).eval()
        tok = self.proc.tokenizer
        self.prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", f"<|{lang}|>", "<|transcribe|>", "<|notimestamps|>"])
        self.eot = tok.eos_token_id

    @torch.no_grad()
    def transcribe(self, audios, max_new=200, tokens=False):
        """-> (texts, mean log-probs); with tokens=True also [[token id, log-prob]] per clip."""
        feats = self.proc.feature_extractor([np.asarray(a, dtype=np.float32) for a in audios], sampling_rate=16000,
                                            return_tensors="pt").input_features.to(self.dev).half()
        m = self.model.model
        enc = m.encoder(feats).last_hidden_state
        B = feats.shape[0]
        step_in = torch.tensor([self.prefix] * B, device=self.dev)
        out = [[] for _ in range(B)]; lp = [[] for _ in range(B)]; done = [False] * B
        past = None
        for _ in range(max_new):
            o = m.decoder(input_ids=step_in, encoder_hidden_states=enc, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logp = torch.log_softmax(self.model.proj_out(o.last_hidden_state[:, -1]).float(), -1)
            nxt = logp.argmax(-1).tolist(); best = logp.max(-1).values.tolist()
            for i in range(B):
                if not done[i]:
                    lp[i].append(best[i])
                    if nxt[i] == self.eot: done[i] = True
                    else: out[i].append(nxt[i])
            if all(done): break
            step_in = torch.tensor(nxt, device=self.dev)[:, None]
        texts = self.proc.tokenizer.batch_decode(out, skip_special_tokens=True)
        means = [sum(x) / len(x) if x else None for x in lp]
        if tokens:
            return [t.strip() for t in texts], means, [list(zip(o, l)) for o, l in zip(out, lp)]
        return [t.strip() for t in texts], means
