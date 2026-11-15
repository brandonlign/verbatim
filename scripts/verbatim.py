"""Verbatim check: pick the transcript that keeps the speaker's language mix, and flag lines to re-check.

Signals (all computed from two Whisper decodes of the same audio):
  disagree   token error rate between the auto decode and the forced+prompted decode
  lang_shift auto decode detected a language other than the primary one (e.g. 'en' or 'ja' for Mandarin speech)
  odd_script output contains characters from a third script (Cyrillic, kana, Hangul...)
  rate       tokens per second of audio (too high = likely hallucination)
  logprob    mean token log-probability of the chosen decode
"""
import re
from opencc import OpenCC

_t2s = OpenCC("t2s")
CJK = r"一-鿿"

def toks(text):
    text = _t2s.convert(str(text or "").lower())
    return re.findall(rf"[{CJK}]|[a-z0-9'\u00e0-\u00ff]+", text)

def edit_rate(a, b):
    a, b = toks(a), toks(b)
    if not a and not b:
        return 0.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m] / max(n, m)

def odd_script(text):
    return bool(re.search(r"[Ѐ-ӿ぀-ヿ가-힯֐-ۿ]", str(text or "")))

def features(rec, primary="zh", a="auto", b="prompt_zh"):
    ta, tb = rec.get(a, ""), rec.get(b, "")
    dur = max(rec.get("dur", 1.0), 0.3)
    return dict(
        disagree=edit_rate(ta, tb),
        lang_shift=(rec.get(a + "_lang") not in (None, primary)),
        odd_script=odd_script(ta) or odd_script(tb),
        rate=len(toks(tb)) / dur,
        logprob=rec.get(b + "_avg_logprob") if rec.get(b + "_avg_logprob") is not None else 0.0,
    )

DEFAULT = dict(disagree=0.35, rate=7.0, logprob=-0.9)

def repetition(text):
    w = toks(text)
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return max([g.count(x) for x in set(g)] or [0])

def broken(rec, key):
    """Decode looks broken: loops, impossible word rate, wrong script, or (nearly) empty vs the auto decode."""
    t = rec.get(key, "") or ""
    n, dur = len(toks(t)), max(rec.get("dur", 1.0), 0.3)
    n_auto = len(toks(rec.get("auto", "")))
    return (odd_script(t) or repetition(t) >= 3 or n / dur > 8
            or (rec.get(key + "_compression") or 0) > 2.4 or (key != "auto" and n < 0.5 * n_auto))

def choose_v2(rec, candidates=("prompt_zh", "auto"), margin=0.15):
    """v2 transcript choice: drop broken decodes, then take the most confident one.
    The first candidate (the forced, prompted decode) wins ties within `margin` log-prob."""
    ok = [k for k in candidates if k in rec and not broken(rec, k)] or [k for k in candidates if k in rec]
    lp = lambda k: rec.get(k + "_avg_logprob") if rec.get(k + "_avg_logprob") is not None else -9.0
    best = ok[0]
    for k in ok[1:]:
        if lp(k) > lp(best) + margin:
            best = k
    return best

def check(rec, th=DEFAULT, primary="zh", a="auto", b="prompt_zh"):
    """Return (chosen_transcript, flagged, reasons)."""
    f = features(rec, primary, a, b)
    reasons = []
    if f["disagree"] > th["disagree"]:
        reasons.append("the two decodes disagree")
    if f["lang_shift"]:
        reasons.append("auto-detect heard a different language")
    if f["odd_script"]:
        reasons.append("unexpected script in output")
    if f["rate"] > th["rate"]:
        reasons.append("too many words for the audio length")
    if f["logprob"] < th["logprob"]:
        reasons.append("low model confidence")
    chosen = rec.get(b, "")
    na, nb = len(toks(rec.get(a, ""))), len(toks(chosen))
    if nb < 0.5 * na:
        # the forced decode lost most of the content (often comes back empty): keep the auto decode
        chosen = rec.get(a, "")
        reasons.append("forced decode came back nearly empty")
    elif odd_script(chosen) and not odd_script(rec.get(a, "")):
        chosen = rec.get(a, "")
    return chosen, bool(reasons), reasons


DEFAULT_V2 = dict(disagree=0.35, rate=7.0, logprob=-0.9)

def check_v2(rec, th=DEFAULT_V2, primary="zh", cands=None):
    """v2: choose among three decodes (forced primary + prompt, auto, forced English + prompt), then flag.
    `cands` overrides the candidate list (first = preferred), e.g. to add the fine-tuned decode 'ft'."""
    p = f"prompt_{primary}"
    cands = tuple(cands) if cands else (p, "auto", "prompt_en")
    key = choose_v2(rec, cands, margin=th.get("margin", 0.15))
    chosen = rec.get(key, "")
    reasons = []
    others = [k for k in cands if k != key and k in rec and not broken(rec, k)]
    dis = min([edit_rate(chosen, rec[k]) for k in others] or [0.0])
    if dis > th["disagree"]:
        reasons.append("the decodes disagree")
    if rec.get("auto_lang") not in (None, primary, "en"):
        reasons.append("auto-detect heard a third language")
    if odd_script(chosen):
        reasons.append("unexpected script in output")
    if len(toks(chosen)) / max(rec.get("dur", 1.0), 0.3) > th["rate"]:
        reasons.append("too many words for the audio length")
    lp = rec.get(key + "_avg_logprob")
    if lp is not None and lp < th["logprob"]:
        reasons.append("low model confidence")
    if sum(broken(rec, k) for k in cands if k in rec) >= 2:
        reasons.append("most decodes look broken")
    return chosen, bool(reasons), reasons, key
