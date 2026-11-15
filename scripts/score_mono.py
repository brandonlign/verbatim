"""Single-language speech: does the fine-tuned system hurt clips with no language mixing?
Scores plain Whisper vs frozen v3 on ASCEND clips labeled zh or en, then on all speech (with the mixed clips).

usage: score_mono.py <mono_run.jsonl> <thresholds_v3_frozen.json> [<mixed_scored.jsonl>]
"""
import json, random, re, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from verbatim import check_v2, edit_count, toks

CJK = re.compile(r"[一-鿿]")
recs = [json.loads(l) for l in open(sys.argv[1])]
th = json.load(open(sys.argv[2]))
for r in recs:
    r["v3"], r["v3_flag"], _, r["v3_key"] = check_v2(r, th, cands=th["cands"])

def mer(rs, key):
    c = [edit_count(r["ref"], r[key]) for r in rs]
    return sum(e for e, _ in c) / max(1, sum(n for _, n in c))

def boot(rs, n=2000, seed=0):
    c = [(edit_count(r["ref"], r["auto"]), edit_count(r["ref"], r["v3"])) for r in rs]
    rng = random.Random(seed); d = []
    for _ in range(n):
        s = [rng.choice(c) for _ in c]; N = sum(x[0][1] for x in s)
        d.append((sum(x[0][0] for x in s) - sum(x[1][0] for x in s)) / N)
    d.sort(); return d[int(.025 * n)], d[int(.975 * n)]

def wrong_language(r, key):
    """Output mostly in the other language: English speech written in Chinese characters, or the reverse."""
    t = toks(r[key])
    if len(t) < 2: return False
    zh = sum(bool(CJK.match(x)) for x in t) / len(t)
    return zh > 0.5 if r["lang"] == "en" else zh < 0.5

print(f"{sys.argv[1].rsplit('/',1)[-1]}: {len(recs)} single-language clips, thresholds frozen {th['frozen_at']}")
for lang in ("zh", "en"):
    rs = [r for r in recs if r["lang"] == lang]
    lo, hi = boot(rs)
    print(f"  {lang}: n={len(rs)}  MER plain {mer(rs,'auto'):.1%} -> v3 {mer(rs,'v3'):.1%} (difference CI {lo:+.1%} to {hi:+.1%})"
          f" | wrong-language output: plain {sum(wrong_language(r,'auto') for r in rs)}, v3 {sum(wrong_language(r,'v3') for r in rs)}"
          f" | v3 flags {sum(r['v3_flag'] for r in rs)/len(rs):.0%} | v3 chose ft on {sum(r['v3_key']=='ft' for r in rs)}")
    by = {}
    for r in rs: by.setdefault(r["speaker"], []).append(r)
    print("     by speaker:", "; ".join(f"{k}: {mer(v,'auto'):.1%} -> {mer(v,'v3'):.1%} (n={len(v)})" for k, v in sorted(by.items())))
if len(sys.argv) > 3:
    mixed = [json.loads(l) for l in open(sys.argv[3])]
    for r in mixed: r["v3"] = check_v2(r, th, cands=th["cands"])[0]
    allr = recs + mixed; lo, hi = boot(allr)
    print(f"  all speech (mixed + single-language, n={len(allr)}): MER plain {mer(allr,'auto'):.1%} -> v3 {mer(allr,'v3'):.1%} (difference CI {lo:+.1%} to {hi:+.1%})")
