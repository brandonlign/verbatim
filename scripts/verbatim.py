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

def edit_count(ref, hyp):
    """(edits, reference tokens): substitutions + deletions + insertions. Used for reported MER."""
    a, b = toks(ref), toks(hyp)
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[len(b)], len(a)

def edit_rate(a, b):
    """Symmetric distance in [0, 1] (edits / longer length). Used only as a flag signal, not for reported MER."""
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
    dur = max(rec.get("dur", 1.0), 0.3)
    def looks_broken(k):
        s = rec.get(k, "") or ""
        return (odd_script(s) or repetition(s) >= 3 or len(toks(s)) / dur > 8
                or (rec.get(k + "_compression") or 0) > 2.4)
    if looks_broken(key):
        return True
    # "nearly empty" is judged against the auto decode, but only when the auto decode isn't itself a loop
    if key != "auto" and not looks_broken("auto"):
        return len(toks(t)) < 0.5 * len(toks(rec.get("auto", "")))
    return False

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


# ---------------------------------------------------------------- v4: learned risk score + word-level marks
import math

def pieces_from_ids(id_lps, tokenizer):
    """[[token id, log-prob]] -> [(text piece, log-prob)]. A character split across tokens takes the lowest log-prob."""
    out, ids, prev, carry = [], [], "", 0.0
    for i, lp in id_lps:
        ids.append(i)
        cur = tokenizer.decode(ids, skip_special_tokens=True)
        carry = min(carry, lp)
        if cur.endswith("�") or len(cur) <= len(prev):
            continue  # incomplete character: wait for the next token
        out.append((cur[len(prev):], carry)); prev, carry = cur, 0.0
    return out

def token_units(pieces):
    """[(text piece, log-prob)] -> (text, [(unit, lowest log-prob, start, end)]) with units as in toks()."""
    text = "".join(p for p, _ in pieces)
    lps = [l for p, l in pieces for _ in p]
    norm = _t2s.convert(text.lower())
    if len(norm) != len(text):
        norm = text.lower() if len(text.lower()) == len(text) else text
    units = [(m.group(), min(lps[m.start():m.end()]), m.start(), m.end())
             for m in re.finditer(rf"[{CJK}]|[a-z0-9'à-ÿ]+", norm)]
    return text, units

def wrong_units(ref, hyp_units):
    """For each transcript unit: True if it is a substitution or insertion against the reference."""
    a, b = toks(ref), list(hyp_units)
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    wrong, i, j = [False] * m, n, m
    while j > 0:
        if i > 0 and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            wrong[j - 1] = a[i - 1] != b[j - 1]; i -= 1; j -= 1
        elif D[i][j] == D[i][j - 1] + 1:
            wrong[j - 1] = True; j -= 1
        else:
            i -= 1
    return wrong

_EN = re.compile(r"[a-z]")
FEATURE_GROUPS = dict(
    base=["dis_zh", "dis_auto", "dis_en", "logprob", "rate", "logdur", "auto_en", "logn"],
    token=["min_tok", "low_share"],
    translate=["tr_match", "tr_sim"],
)

def v4_features(rec, chosen, key, primary="zh"):
    """Signals for the v4 risk score. `rec['ft_pieces']` (token pieces of the fine-tuned decode) is optional."""
    p = f"prompt_{primary}"
    dur = max(rec.get("dur", 1.0), 0.3)
    t = toks(chosen)
    lp = rec.get(key + "_avg_logprob")
    lp = -1.0 if lp is None else lp
    f = dict(dis_zh=edit_rate(chosen, rec.get(p, "")), dis_auto=edit_rate(chosen, rec.get("auto", "")),
             dis_en=edit_rate(chosen, rec.get("prompt_en", "")), logprob=lp, rate=len(t) / dur, logdur=math.log(dur),
             auto_en=float(rec.get("auto_lang") == "en"), logn=math.log1p(len(t)))
    if key == "ft" and rec.get("ft_pieces"):
        u = token_units(rec["ft_pieces"])[1]
        f["min_tok"] = min([x[1] for x in u] or [lp]); f["low_share"] = sum(x[1] < -1.0 for x in u) / max(1, len(u))
    else:
        f["min_tok"], f["low_share"] = lp, 0.0
    tr = rec.get("translate")
    if tr is None:
        f["tr_match"], f["tr_sim"] = 0.0, 0.0
    else:
        trs, zen = set(toks(tr)), {w for w in toks(rec.get(p, "")) if _EN.match(w)}
        f["tr_match"] = sum(1 for w in t if _EN.match(w) and w not in zen and w in trs) / max(1, len(t))
        f["tr_sim"] = 1.0 - edit_rate(chosen, tr)
    return f

def risk_v4(rec, model, primary="zh"):
    """(transcript, risk in [0,1], flagged, chosen decode). `model` is the frozen v4 JSON."""
    chosen, _, _, key = check_v2(rec, model["v3"], primary=primary, cands=model["v3"]["cands"])
    f = v4_features(rec, chosen, key, primary)
    z = model["bias"] + sum(w * (f[n] - mu) / sd for n, w, mu, sd in zip(model["features"], model["weights"], model["mean"], model["std"]))
    risk = 1 / (1 + math.exp(-z))
    return chosen, risk, risk >= model["threshold"], key
