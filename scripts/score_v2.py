"""Score plain Whisper vs Verbatim v2 (frozen thresholds) on a run file, by language pair.

usage: score_v2.py <run.jsonl> <thresholds_v2_frozen.json> <primary: zh|es> [--extra KEY] [--out scored.jsonl]

Works on ASCEND run files (ref only) and ServiceNow run files (ref + per-word language tags).
--extra KEY adds another decode (e.g. 'ft' from the fine-tuned model) as a standalone system.
"""
import argparse, json, math, re, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verbatim import check_v2, edit_count, edit_rate, toks
from taxonomy import classify as classify_zh

CJK = re.compile(r"[一-鿿]")

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def ref_lang_tokens(r, primary):
    """Reference tokens split by language: from word tags when present, else by script."""
    if r.get("word_languages"):
        out = {"en": [], primary: []}
        for w, l in zip(r["words"], r["word_languages"]):
            out["en" if l.upper() == "EN" else primary].extend(toks(w))
        return out
    t = toks(r["ref"])
    return {"en": [x for x in t if not CJK.match(x)], primary: [x for x in t if CJK.match(x)]}

def recall(ref_toks, hyp):
    if not ref_toks: return None
    h = {}
    for x in toks(hyp): h[x] = h.get(x, 0) + 1
    hit = 0
    for x in ref_toks:
        if h.get(x, 0) > 0: hit += 1; h[x] -= 1
    return hit / len(ref_toks)

def classify(r, hyp, primary):
    if primary == "zh":
        return classify_zh(r["ref"], hyp)
    rt = ref_lang_tokens(r, primary); n_ref = len(rt["en"]) + len(rt[primary]); n_h = len(toks(hyp))
    if n_h > 3 * max(n_ref, 4): return "hallucination"
    pr, er = recall(rt[primary], hyp), recall(rt["en"], hyp)
    if pr is not None and pr < 0.4 and n_h >= 0.5 * n_ref and (er is None or er >= 0.5):
        return f"translated_{primary}_to_en"
    if er is not None and er < 0.4 and n_h >= 0.5 * n_ref and (pr is None or pr >= 0.6):
        return f"translated_en_to_{primary}"
    if er == 0: return "dropped_en"
    if pr == 0: return f"dropped_{primary}"
    return "ok_or_minor"

def is_bad(r, hyp, primary):
    return classify(r, hyp, primary) != "ok_or_minor" or edit_rate(r["ref"], hyp) > 0.5

def mer(recs, key):
    """Corpus MER: total edits / total reference tokens."""
    c = [edit_count(r["ref"], r[key]) for r in recs]
    return sum(e for e, _ in c) / sum(n for _, n in c)

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("th"); ap.add_argument("primary")
ap.add_argument("--extra"); ap.add_argument("--out")
ap.add_argument("--cands", help="comma-separated candidate decodes for v2, preferred first")
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.run)]
th = json.load(open(a.th)); P = a.primary
for r in recs:
    r["v2"], r["v2_flag"], r["v2_reasons"], r["v2_key"] = check_v2(r, th, primary=P, cands=a.cands.split(",") if a.cands else None)

systems = ["auto", f"prompt_{P}", "prompt_en"] + ([a.extra] if a.extra else []) + ["v2"]
systems = [s for s in systems if all(s in r for r in recs)]
n = len(recs)
print(f"{a.run.rsplit('/',1)[-1]}: n = {n}, primary = {P}, thresholds frozen {th['frozen_at']}\n")
print(f"{'system':12s} {'MER':>6s} {'EN-word recall':>15s} {P.upper()+'-word recall':>15s} {'translated->en':>15s} {'all bad':>8s}")
for s in systems:
    er = [x for x in (recall(ref_lang_tokens(r, P)["en"], r[s]) for r in recs) if x is not None]
    pr = [x for x in (recall(ref_lang_tokens(r, P)[P], r[s]) for r in recs) if x is not None]
    tr = sum(classify(r, r[s], P) == f"translated_{P}_to_en" for r in recs)
    bad = sum(is_bad(r, r[s], P) for r in recs)
    print(f"{s:12s} {mer(recs, s):6.3f} {sum(er)/len(er):15.3f} {sum(pr)/len(pr):15.3f} {tr:>8d} ({tr/n:4.1%}) {bad:>8d}")

tr_auto = [r for r in recs if classify(r, r["auto"], P) == f"translated_{P}_to_en"]
lo, hi = wilson(len(tr_auto), n)
print(f"\nPlain-Whisper silent translations ({P}->en): {len(tr_auto)}/{n} = {len(tr_auto)/n:.1%}  95% CI {lo:.1%}-{hi:.1%}")
print(f"  flagged by v2: {sum(r['v2_flag'] for r in tr_auto)}/{len(tr_auto)}"
      f"   fixed in v2 transcript: {sum(classify(r, r['v2'], P) != f'translated_{P}_to_en' for r in tr_auto)}/{len(tr_auto)}")
tp = sum(r["v2_flag"] and is_bad(r, r["v2"], P) for r in recs); fl = sum(r["v2_flag"] for r in recs)
bad = sum(is_bad(r, r["v2"], P) for r in recs)
print(f"v2 flag on its own transcript: precision {tp}/{fl} = {tp/max(fl,1):.2f}, recall {tp}/{bad} = {tp/max(bad,1):.2f}, flags {fl/n:.0%} of lines")
print("v2 chose:", {k: sum(r["v2_key"] == k for r in recs) for k in sorted({r['v2_key'] for r in recs})})
if a.out:
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
