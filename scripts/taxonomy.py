"""Classify each transcript's failure mode: ok / silent translation / dropped language / hallucination / other."""
import re, sys
import pandas as pd
from opencc import OpenCC

t2s = OpenCC("t2s")

def toks(t):
    t = t2s.convert(str(t).lower()) if isinstance(t, str) else ""
    return re.findall(r"[一-鿿]|[a-z0-9']+", t)

FILLERS = {"呃", "嗯", "啊", "哦", "的", "就", "是", "那", "个"}

def classify(ref, hyp):
    r, h = toks(ref), toks(hyp)
    r_zh = [c for c in r if re.match(r"[一-鿿]", c) and c not in FILLERS]
    r_en = [w for w in r if re.match(r"[a-z]", w)]
    h_zh = [c for c in h if re.match(r"[一-鿿]", c)]
    h_en = [w for w in h if re.match(r"[a-z]", w)]
    if len(h) > 3 * max(len(r), 4):
        return "hallucination"
    zh_rec = sum(c in set(h_zh) for c in r_zh) / len(r_zh) if r_zh else 1
    en_rec = sum(w in set(h_en) for w in r_en) / len(r_en) if r_en else 1
    extra_en = len([w for w in h_en if w not in set(r_en)])
    extra_zh = len([c for c in h_zh if c not in set(r_zh)])
    if r_zh and zh_rec < 0.4 and extra_en >= 2:
        return "translated_zh_to_en"
    if r_en and en_rec < 0.4 and extra_zh >= 2:
        return "translated_en_to_zh"
    if r_en and en_rec == 0:
        return "dropped_en"
    if r_zh and zh_rec == 0:
        return "dropped_zh"
    return "ok_or_minor"

if __name__ == "__main__":
    d = pd.read_csv(sys.argv[1])
    for col in sys.argv[2:]:
        c = d.apply(lambda x: classify(x.ref, x[col]), axis=1)
        print(f"\n{col}:")
        print(c.value_counts().to_string())
