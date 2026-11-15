"""Fit the v4 risk score on VALIDATION ONLY and freeze it.

The transcript is the frozen v3 transcript. v4 replaces the yes/no flag rules with one logistic risk score,
and sets a token log-prob cut-off for marking single words.

usage: tune_v4.py <asr_val_v4.jsonl> <thresholds_v3_frozen.json> <tokenizer_dir> <out thresholds_v4_frozen.json>
"""
import datetime, json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verbatim import (check_v2, edit_count, v4_features, pieces_from_ids, token_units, wrong_units, FEATURE_GROUPS)

MAJOR = 0.20  # target: 20% or more of the reference tokens wrong

def prepare(recs, th3, tokenizer):
    rows = []
    for r in recs:
        if r.get("ft_tokens") is not None:
            r["ft_pieces"] = pieces_from_ids(r["ft_tokens"], tokenizer)
        chosen, _, _, key = check_v2(r, th3, cands=th3["cands"])
        e, n = edit_count(r["ref"], chosen)
        rows.append(dict(rec=r, chosen=chosen, key=key, f=v4_features(r, chosen, key), y=e / max(1, n) >= MAJOR))
    return rows

def fit(X, y, l2=1.0):
    """L2 logistic regression by Newton steps (features already standardized)."""
    Xb = np.hstack([X, np.ones((len(X), 1))]); w = np.zeros(Xb.shape[1])
    R = np.eye(Xb.shape[1]) * l2; R[-1, -1] = 0
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + R @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8: break
    return w

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))

def matrix(rows, names):
    return np.array([[r["f"][n] for n in names] for r in rows], float)

def loo_scores(rows, names):
    X, y = matrix(rows, names), np.array([r["y"] for r in rows], float)
    out = np.zeros(len(rows))
    for i in range(len(rows)):
        m = np.arange(len(rows)) != i
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-6
        w = fit((X[m] - mu) / sd, y[m])
        out[i] = np.append((X[i] - mu) / sd, 1) @ w
    return out

if __name__ == "__main__":
    from transformers import WhisperTokenizer
    val_path, th3_path, tok_dir, out_path = sys.argv[1:5]
    th3 = json.load(open(th3_path)); tok = WhisperTokenizer.from_pretrained(tok_dir)
    rows = prepare([json.loads(l) for l in open(val_path)], th3, tok)
    y = np.array([r["y"] for r in rows])
    print(f"validation lines {len(rows)}, lines with {MAJOR:.0%}+ tokens wrong: {y.sum()} ({y.mean():.0%})")

    names = list(FEATURE_GROUPS["base"]); best = auc(y, loo_scores(rows, names)); kept = ["base"]
    print(f"leave-one-out AUC, base signals: {best:.3f}")
    for g in ("token",):  # the translate-match group was dropped: turbo does not translate (see NOTES_INTEGRITY.md)
        a = auc(y, loo_scores(rows, names + FEATURE_GROUPS[g]))
        keep = a > best
        print(f"  + {g}: {a:.3f} -> {'kept' if keep else 'not kept'}")
        if keep: names += FEATURE_GROUPS[g]; best = a; kept.append(g)

    s = 1 / (1 + np.exp(-loo_scores(rows, names)))  # threshold picked on held-out (leave-one-out) scores
    cands = sorted(set(np.round(s, 3)))
    def stats(t):
        f = s >= t; tp = (f & y).sum()
        p = tp / max(1, f.sum()); rc = tp / y.sum()
        return 2 * p * rc / max(1e-9, p + rc), p, rc, f.mean()
    thr = max(cands, key=lambda t: stats(t)[0]); f1, p, rc, fr = stats(thr)
    print(f"threshold {thr:.3f}: validation precision {p:.2f}, recall {rc:.2f}, flags {fr:.0%} of lines")

    # word marks: token log-prob cut-off, chosen for F1 on validation words of the fine-tuned decode
    lps, wrong = [], []
    for r in rows:
        if r["key"] == "ft" and r["rec"].get("ft_pieces"):
            u = token_units(r["rec"]["ft_pieces"])[1]
            w = wrong_units(r["rec"]["ref"], [x[0] for x in u])
            lps += [x[1] for x in u]; wrong += w
    lps, wrong = np.array(lps), np.array(wrong, bool)
    def wstats(c):
        m = lps < c; tp = (m & wrong).sum(); p = tp / max(1, m.sum()); rc = tp / max(1, wrong.sum())
        return 2 * p * rc / max(1e-9, p + rc), p, rc, m.mean()
    cut = max([-x / 20 for x in range(2, 81)], key=lambda c: wstats(c)[0]); wf1, wp, wr, wm = wstats(cut)
    print(f"words: {len(lps)}, wrong {wrong.mean():.1%}; AUC of token log-prob {auc(wrong, -lps):.3f}")
    print(f"word cut-off {cut:.2f}: validation precision {wp:.2f}, recall {wr:.2f}, marks {wm:.1%} of words")

    X = matrix(rows, names); mu, sd = X.mean(0), X.std(0) + 1e-6
    w = fit((X - mu) / sd, y.astype(float))
    model = dict(features=names, groups=kept, weights=[float(x) for x in w[:-1]], bias=float(w[-1]), mean=[float(x) for x in mu], std=[float(x) for x in sd],
                 threshold=float(thr), word_logprob=float(cut), target=f"{MAJOR:.0%}+ reference tokens wrong",
                 v3={k: th3[k] for k in ("disagree", "rate", "logprob", "margin", "cands")},
                 frozen_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), tuned_on=val_path.rsplit("/", 1)[-1],
                 validation_stats=dict(loo_auc=float(best), precision=float(p), recall=float(rc), flag_rate=float(fr),
                                       word_precision=float(wp), word_recall=float(wr), word_mark_rate=float(wm)))
    json.dump(model, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in model.items() if k not in ("mean", "std")}, indent=1))
