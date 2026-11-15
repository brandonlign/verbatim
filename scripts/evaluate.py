"""Score systems + the Verbatim checker on a run file.

tune:  evaluate.py tune <asr_val.jsonl> <thresholds.json>   (grid search on validation, then writes frozen thresholds)
test:  evaluate.py test <asr_test.jsonl> <thresholds.json> [<out_prefix>]
"""
import json, sys, itertools, datetime, random
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verbatim import check, edit_rate, features
from taxonomy import classify

SYSTEMS = ["auto", "force_zh", "prompt_zh"]
BAD = {"translated_zh_to_en", "translated_en_to_zh", "dropped_en", "dropped_zh", "hallucination"}

def load(p):
    return [json.loads(l) for l in open(p)]

def corpus_mer(recs, key):
    """Corpus MER: total edits / total reference tokens."""
    from verbatim import edit_count
    c = [edit_count(r["ref"], r[key]) for r in recs]
    return sum(e for e, _ in c) / sum(n for _, n in c)

def is_bad(ref, hyp):
    return classify(ref, hyp) in BAD or edit_rate(ref, hyp) > 0.5

def detector_stats(recs, th):
    tp = fp = fn = tn = 0
    for r in recs:
        chosen, flagged, _ = check(r, th)
        bad = is_bad(r["ref"], chosen)
        tp += bad and flagged; fp += (not bad) and flagged; fn += bad and not flagged; tn += (not bad) and not flagged
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, precision=prec, flag_rate=(tp + fp) / len(recs))

def boot_ci(vals, n=2000, seed=0):
    rng = random.Random(seed); vals = list(vals)
    if not vals: return (float("nan"), float("nan"))
    ms = sorted(np.mean([rng.choice(vals) for _ in vals]) for _ in range(n))
    return (ms[int(.025 * n)], ms[int(.975 * n)])

def tune(path, out):
    recs = load(path)
    best = None
    for d, rt, lp in itertools.product([.2, .25, .3, .35, .4, .5, .6], [5, 6, 7, 8, 10], [-0.6, -0.7, -0.8, -0.9, -1.0, -1.2]):
        th = dict(disagree=d, rate=rt, logprob=lp)
        s = detector_stats(recs, th)
        # goal: best balance of catching bad lines and not flagging good ones (F1)
        p, rc = s["precision"], s["recall"]
        f1 = 2 * p * rc / (p + rc) if p + rc else 0
        score = (round(f1, 4), -s["flag_rate"])
        if best is None or score > best[0]:
            best = (score, th, s)
    _, th, s = best
    th["frozen_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    th["tuned_on"] = path.rsplit("/", 1)[-1]
    th["validation_stats"] = s
    json.dump(th, open(out, "w"), indent=2)
    print(json.dumps(th, indent=2))

def test(path, thpath, prefix=None):
    recs = load(path); th = json.load(open(thpath))
    print(f"n = {len(recs)} mixed utterances; thresholds frozen at {th['frozen_at']}")
    for r in recs:
        r["verbatim"], r["flagged"], r["reasons"] = check(r, th)
    print("\nCorpus mixed error rate:")
    for k in SYSTEMS + ["verbatim"]:
        print(f"  {k:10s} {corpus_mer(recs, k):.3f}")
    print("\nFailure types (automatic classifier):")
    rows = {}
    for k in SYSTEMS + ["verbatim"]:
        c = {}
        for r in recs:
            t = classify(r["ref"], r[k]); c[t] = c.get(t, 0) + 1
        rows[k] = c
    types = sorted({t for c in rows.values() for t in c})
    print("  " + " ".join(f"{t[:18]:>19s}" for t in types))
    for k, c in rows.items():
        print(f"  {k:10s}" + "".join(f"{c.get(t,0):>19d}" for t in types))
    st = [classify(r["ref"], r["auto"]).startswith("translated_zh") for r in recs]
    n_st = sum(st)
    fixed = sum(1 for r, s in zip(recs, st) if s and not classify(r["ref"], r["verbatim"]).startswith("translated"))
    flagged = sum(1 for r, s in zip(recs, st) if s and r["flagged"])
    print(f"\nPlain-Whisper silent translations (Chinese -> English): {n_st}/{len(recs)} = {n_st/len(recs):.1%}"
          f"  95% CI {tuple(round(x,3) for x in boot_ci([float(s) for s in st]))}")
    print(f"  fixed by Verbatim's chosen transcript: {fixed}/{n_st}")
    print(f"  flagged by Verbatim: {flagged}/{n_st}")
    s = detector_stats(recs, th)
    print(f"\nDetector on Verbatim's own output: recall {s['recall']:.2f}, precision {s['precision']:.2f}, flags {s['flag_rate']:.0%} of lines  ({s})")
    if prefix:
        with open(prefix + "_scored.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    {"tune": tune, "test": test}[sys.argv[1]](*sys.argv[2:])
