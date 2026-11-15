"""Standard corpus-level mixed error rate: total edits (substitutions + deletions + insertions)
divided by total reference tokens (Chinese characters + English words, OpenCC t2s)."""
import json, random, sys
sys.path.insert(0, "scripts")
from verbatim import check_v2, edit_count

edits = edit_count

def load(p): return [json.loads(l) for l in open(p)]

def table(recs, systems):
    E = {k: [edits(r["ref"], f(r)) for r in recs] for k, f in systems.items()}
    out = {}
    for k, v in E.items():
        out[k] = sum(e for e, _ in v) / sum(n for _, n in v)
    return E, out

def boot(E, a, b, n=2000, seed=0):
    rng = random.Random(seed); idx = range(len(E[a])); diffs, rels = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        N = sum(E[a][i][1] for i in s)
        ea = sum(E[a][i][0] for i in s) / N; eb = sum(E[b][i][0] for i in s) / N
        diffs.append(ea - eb); rels.append((ea - eb) / ea)
    diffs.sort(); rels.sort()
    q = lambda x: (x[int(.025 * n)], x[int(.975 * n)])
    return q(diffs), q(rels)

th2 = json.load(open("data/thresholds_v2_frozen.json"))
th3 = json.load(open("data/thresholds_v3_frozen.json"))

print("== ASCEND test (369) ==")
t1 = {r["id"]: r for r in load("data/test_scored.jsonl")}
t3 = load("data/test_v3_scored.jsonl")
sys_t = {
    "plain (auto)": lambda r: r["auto"],
    "Mandarin forced + prompt": lambda r: r["prompt_zh"],
    "English forced + prompt": lambda r: r["prompt_en"],
    "v1": lambda r: t1[r["id"]]["verbatim"],
    "v2": lambda r: check_v2(r, th2, "zh")[0],
    "fine-tuned alone": lambda r: r["ft"],
    "v3": lambda r: r["v2"],
}

E, m = table(t3, sys_t)
for k, v in m.items(): print(f"  {k:28s} {v:.1%}")
(d, rel) = boot(E, "plain (auto)", "v3")
print(f"  v3 vs plain: {m['plain (auto)']-m['v3']:.1%} points (CI {d[0]:.1%}-{d[1]:.1%}), relative {1-m['v3']/m['plain (auto)']:.0%} (CI {rel[0]:.0%}-{rel[1]:.0%})")
better = sum(a[0] / max(a[1],1) > b[0] / max(b[1],1) for a, b in zip(E["plain (auto)"], E["v3"]))
worse = sum(a[0] / max(a[1],1) < b[0] / max(b[1],1) for a, b in zip(E["plain (auto)"], E["v3"]))
print(f"  per line: v3 better {better}, worse {worse}, same {len(t3)-better-worse}")

print("== ASCEND validation (210) ==")
v = load("data/asr_val_ft.jsonl")
_, mv = table(v, {"plain": lambda r: r["auto"], "v2": lambda r: check_v2(r, th2, "zh")[0], "fine-tuned": lambda r: r["ft"],
                  "v3": lambda r: check_v2(r, th3, "zh", cands=th3["cands"])[0]})
for k, x in mv.items(): print(f"  {k:28s} {x:.1%}")

print("== ServiceNow zh (294) ==")
z = load("data/sn_zh_v3_scored.jsonl")
_, mz = table(z, {"plain": lambda r: r["auto"], "v2": lambda r: check_v2(r, th2, "zh")[0], "v3": lambda r: r["v2"]})
for k, x in mz.items(): print(f"  {k:28s} {x:.1%}")
print("== ServiceNow es (259) ==")
e = load("data/sn_es_scored.jsonl")
_, me = table(e, {"plain": lambda r: r["auto"], "v2": lambda r: r["v2"]})
for k, x in me.items(): print(f"  {k:28s} {x:.1%}")
