"""Tune v3 flag thresholds on validation (F1), freeze to JSON.
v3 = fine-tuned decode 'ft' preferred, base Whisper decodes as cross-checks."""
import json, sys, itertools, datetime
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verbatim import check_v2
from evaluate import is_bad
CANDS = ["ft", "prompt_zh", "auto", "prompt_en"]
recs = [json.loads(l) for l in open(sys.argv[1])]
best = None
for d, rt, lp in itertools.product([.1, .15, .2, .25, .3, .35, .4, .5, .6, .8], [5, 6, 7, 8, 10],
                                   [-0.1, -0.15, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8, -1.0, -9]):
    th = dict(disagree=d, rate=rt, logprob=lp, margin=0.15)
    tp = fp = fn = 0
    for r in recs:
        c, f, _, _ = check_v2(r, th, cands=CANDS)
        b = is_bad(r["ref"], c); tp += b and f; fp += (not b) and f; fn += b and not f
    p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0
    if best is None or f1 > best[0]:
        best = (f1, th, dict(tp=tp, fp=fp, fn=fn, precision=p, recall=rc, flag_rate=(tp + fp) / len(recs)))
f1, th, st = best
th.update(frozen_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), tuned_on=sys.argv[1].rsplit("/", 1)[-1],
          cands=CANDS, validation_stats=st)
json.dump(th, open(sys.argv[2], "w"), indent=2); print(json.dumps(th, indent=2))
