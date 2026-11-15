"""Score the frozen v4 risk score and word marks on a run file (references, and human labels when given).

usage: score_v4.py <run_v4.jsonl> <thresholds_v4_frozen.json> <tokenizer_dir> [--labels labels_v3.jsonl --label-set label_set2.jsonl] [--frozen-run test_v3_scored.jsonl] [--out scored.jsonl]
"""
import argparse, json, math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verbatim import check_v2, edit_count, risk_v4, pieces_from_ids, token_units, wrong_units
from taxonomy import classify
from tune_v4 import auc, MAJOR

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/max(n,1):.0%} (CI {lo:.0%}-{hi:.0%})"

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("model"); ap.add_argument("tok")
ap.add_argument("--labels"); ap.add_argument("--label-set", default="data/label_set2.jsonl"); ap.add_argument("--frozen-run"); ap.add_argument("--out")
a = ap.parse_args()
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained(a.tok)
M = json.load(open(a.model)); recs = [json.loads(l) for l in open(a.run)]
if a.frozen_run:
    old = {r["id"]: r for r in map(json.loads, open(a.frozen_run))}
    print("fine-tuned text identical to the frozen decode:", sum(r["ft"] == old[r["id"]]["ft"] for r in recs), "of", len(recs))

y, risk, flag, v3flag, wl, ww = [], [], [], [], [], []
for r in recs:
    r["ft_pieces"] = pieces_from_ids(r["ft_tokens"], tok)
    chosen, rk, fl, key = risk_v4(r, M)
    _, f3, _, _ = check_v2(r, M["v3"], cands=M["v3"]["cands"])
    e, n = edit_count(r["ref"], chosen)
    r.update(v4=chosen, v4_risk=rk, v4_flag=bool(fl), v3_flag=bool(f3), v4_key=key, edits=e, ntok=n)
    y.append(e / max(1, n) >= MAJOR); risk.append(rk); flag.append(fl); v3flag.append(f3)
    if key == "ft":
        u = token_units(r["ft_pieces"])[1]
        wl += [x[1] for x in u]; ww += wrong_units(r["ref"], [x[0] for x in u])
y, risk, flag, v3flag = np.array(y), np.array(risk), np.array(flag, bool), np.array(v3flag, bool)
E_ = np.array([r["edits"] for r in recs]); anyerr = E_ > 0

print(f"\n{a.run.rsplit('/',1)[-1]}: {len(recs)} lines, model frozen {M['frozen_at']}")
print(f"Against references: {y.sum()} lines have {MAJOR:.0%}+ tokens wrong; risk score AUC {auc(y, risk):.3f}")
for name, f in (("v3 flag", v3flag), ("v4 flag", flag)):
    print(f"  {name}: flags {f.mean():.0%} of lines | badly wrong lines caught {frac(int((f & y).sum()), int(y.sum()))} | "
          f"flagged lines with any error {frac(int((f & anyerr).sum()), int(f.sum()))} | share of all wrong tokens inside flagged lines {E_[f].sum()/E_.sum():.0%}")
print("  If a reviewer checks the riskiest lines first:")
o = np.argsort(-risk)
for fr in (.1, .2, .3, .4, .5):
    k = int(round(fr * len(recs))); s = o[:k]
    print(f"    top {fr:.0%}: {y[s].sum()/y.sum():.0%} of badly wrong lines, {E_[s].sum()/E_.sum():.0%} of all wrong tokens")

wl, ww = np.array(wl), np.array(ww, bool); m = wl < M["word_logprob"]
print(f"Word marks (cut-off {M['word_logprob']}): {len(wl)} words, {ww.sum()} wrong; token log-prob AUC {auc(ww, -wl):.3f}")
print(f"  marked words that are wrong {frac(int((m & ww).sum()), int(m.sum()))} | wrong words marked {frac(int((m & ww).sum()), int(ww.sum()))} | marks {m.mean():.1%} of words")

if a.labels:
    lab = {}
    for l in map(json.loads, open(a.labels)):
        if l["system"] == "v3" and l["label"] != "skip": lab[l["id"]] = l["label"]
    # lines where v3's output was identical to plain Whisper's were shown once, labeled under "auto"
    same = {s["id"] for s in map(json.loads, open(a.label_set)) if s.get("v3_same_as_auto")}
    for l in map(json.loads, open(a.labels)):
        if l["system"] == "auto" and l["id"] in same and l["label"] != "skip": lab[l["id"]] = l["label"]
    L = [r for r in recs if r["id"] in lab]
    for title, S in (("random lines", [r for r in L if classify(r["ref"], r["auto"]) != "translated_zh_to_en"]), ("all labeled lines", L)):
        err = [lab[r["id"]] != "ok" for r in S]; tr = [lab[r["id"]] == "silent_translation" for r in S]
        print(f"\nHuman labels, {title} ({len(S)}; {sum(err)} with an error, {sum(tr)} translations)")
        for name, k in (("v3 flag", "v3_flag"), ("v4 flag", "v4_flag")):
            f = [r[k] for r in S]
            print(f"  {name}: flags that are real errors {frac(sum(a and b for a, b in zip(f, err)), sum(f))} | "
                  f"errors flagged {frac(sum(a and b for a, b in zip(f, err)), sum(err))} | translations flagged {sum(a and b for a, b in zip(f, tr))}/{sum(tr)}")
        ya = np.array(err); rk = np.array([r["v4_risk"] for r in S])
        print(f"  risk score AUC against human 'has an error': {auc(ya, rk):.3f}")
if a.out:
    with open(a.out, "w") as f:
        for r in recs:
            r.pop("ft_pieces", None); f.write(json.dumps(r, ensure_ascii=False) + "\n")
