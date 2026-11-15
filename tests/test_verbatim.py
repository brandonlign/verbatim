"""Tests for the Verbatim checks, plus regression tests that tie the code to the reported numbers.

run: .venv/bin/python tests/test_verbatim.py   (or: pytest tests/)
"""
import json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from verbatim import broken, check_v2, choose_v2, edit_count, odd_script, toks  # noqa: E402


def load(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "data", name))]


def frozen(name):
    return json.load(open(os.path.join(ROOT, "data", name)))


def mer(recs, hyp):
    e = [edit_count(r["ref"], hyp(r)) for r in recs]
    return sum(x for x, _ in e) / sum(n for _, n in e)


# ---------------------------------------------------------------- units

def test_tokens_split_chinese_characters_and_english_words():
    assert toks("那你有没有什么favourite foods") == ["那", "你", "有", "没", "有", "什", "么", "favourite", "foods"]


def test_tokens_convert_traditional_to_simplified():
    assert toks("個人") == toks("个人")


def test_edit_count_counts_insertions_in_full():
    # a hallucinated tail is 3 insertions against a 2-token reference: MER 150%, not capped
    assert edit_count("hard choice", "hard choice oh i go") == (3, 2)


def test_edit_count_translation_is_heavily_penalised():
    edits, n = edit_count("那你有没有什么favourite foods", "That you have no favorite food?")
    assert n == 9 and edits >= 8


def test_odd_script_detects_third_language():
    assert odd_script("これは日本語") and not odd_script("这是中文 and English")


def test_loop_is_broken():
    rec = {"dur": 3.0, "auto": "go " * 50, "ft": "我们 go"}
    assert broken(rec, "auto")


def test_good_decode_survives_when_auto_loops():
    # regression for test clip 00606: plain Whisper looped, so "nearly empty vs auto" must not apply
    rec = {"dur": 3.0, "auto": "go " * 223, "ft": "那我们就go吧"}
    assert not broken(rec, "ft")
    assert choose_v2(rec, ("ft", "auto")) == "ft"


def test_more_confident_decode_wins_only_beyond_margin():
    rec = {"dur": 2.0, "prompt_zh": "然后当了one semester", "prompt_zh_avg_logprob": -0.30,
           "auto": "and then one semester", "auto_avg_logprob": -0.20}
    assert choose_v2(rec, ("prompt_zh", "auto"), margin=0.15) == "prompt_zh"
    rec["auto_avg_logprob"] = -0.10
    assert choose_v2(rec, ("prompt_zh", "auto"), margin=0.15) == "auto"


# ---------------------------------------------------------------- regression against the reported results

def test_v3_code_reproduces_the_frozen_test_transcripts():
    th3 = frozen("thresholds_v3_frozen.json")
    recs = load("test_v3_scored.jsonl")
    assert len(recs) == 369
    same = sum(check_v2(r, th3, "zh", cands=th3["cands"])[0] == r["v2"] for r in recs)
    assert same == 369


def test_reported_test_mer():
    recs = load("test_v3_scored.jsonl")
    assert round(mer(recs, lambda r: r["auto"]), 3) == 0.233
    assert round(mer(recs, lambda r: r["v2"]), 3) == 0.123


def test_reported_validation_mer_behind_the_adoption_decision():
    th2, th3 = frozen("thresholds_v2_frozen.json"), frozen("thresholds_v3_frozen.json")
    recs = load("asr_val_ft.jsonl")
    assert round(mer(recs, lambda r: r["auto"]), 3) == 0.267
    assert round(mer(recs, lambda r: check_v2(r, th2, "zh")[0]), 3) == 0.212
    assert round(mer(recs, lambda r: check_v2(r, th3, "zh", cands=th3["cands"])[0]), 3) == 0.161


def test_thresholds_were_frozen_in_order():
    t = [frozen(f"thresholds_{v}frozen.json")["frozen_at"] for v in ("", "v2_", "v3_")]
    assert t == sorted(t)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f(); print("ok  ", n)
        except AssertionError as e:
            failed += 1; print("FAIL", n, e)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
