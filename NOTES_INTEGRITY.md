# Integrity log (what was done in what order)

- 2026-11-14 ~12:20 ET: pilot on 80 random ASCEND *test* mixed utterances (auto / force_zh / force_en / prompt_auto / prompt_zh). These 80 are inside the final test set, so the choice of method (forced Chinese + code-switched prompt) was informed by test utterances. Disclosed in RESULTS.md.
- 2026-11-14 ~18:50 ET: while designing the fallback rule, error rates per system grouped by detected language were printed for both validation and test. No threshold was chosen from the test numbers.
- 2026-11-14 18:50:43 ET: checker thresholds tuned on ASCEND validation only (different speakers from test) and frozen in data/thresholds_frozen.json.
- After freeze: test scored once (data/test_report.txt). Hand labels collected blind (systems shuffled and unnamed) on 100 test utterances: all 29 that the automatic classifier marked as Chinese->English translation + 71 random others.
- (Planned teammate recordings were not made; no recorded data is used anywhere.)
- 2026-11-14 ~19:30 ET: hand labels (200 blind transcripts) collected on the test subset; results computed once.
- 2026-11-14 20:19:33 ET: **v2 frozen** (data/thresholds_v2_frozen.json). Three decodes (forced primary language + prompt, auto, forced English + prompt); broken decodes dropped (loops, impossible word rate, wrong script, near-empty); most confident remaining decode chosen (margin 0.15). Rule and flag thresholds chosen on ASCEND validation only.
- v2 is evaluated on sets never used for any design decision: ServiceNow asr_codeswitched en_zh (294) and en_es (259). On ASCEND test it is reported as a second look at an already-used set.
- 2026-11-14 20:30 ET: decoder-only fine-tune of Whisper large-v3-turbo started on ASCEND **train** (8,387 utterances, speakers disjoint from validation and test). Encoder frozen. 800 steps, batch 8, lr 1e-5. Whether to adopt it is decided on validation.

## 2026-11-14 22:27 — ServiceNow en_zh scored with frozen v2 (unseen)
- scripts/score_v2.py written after the v2 freeze; uses the same taxonomy classifier and is_bad rule as evaluate.py (Spanish uses word-tag-based recall).
- Result: plain Whisper zh->en silent translations 0/294 (CI 0–1.3%); v2 picked prompt_zh on all 294; v2 flags 3% of lines. The en->zh proxy over-counts here (brand-name spelling like Workday→"Work Day", Wi‑Fi→WiFi), so MER and zh->en counts are the reported numbers.
- Queued GPU jobs (one at a time): ft on validation → prompt_en on test → ft on test → ft on ServiceNow zh → resume Spanish. ft outputs on test are produced but not scored until adoption is decided on validation.

## 2026-11-14 23:06 — Fine-tune adoption rule (written before seeing any fine-tuned output)
- Fine-tune finished 23:06 (800 steps, ~0.77 epoch of ASCEND train, decoder only).
- Candidates compared on VALIDATION only (asr_val_ft.jsonl):
  A. ft alone;  B. v2 with ft in place of prompt_zh as the preferred decode;  C. v2 with ft added as a 4th candidate.
- Flag thresholds stay the frozen v2 ones (no re-tuning).
- Adopt the option with the lowest validation MER, only if it beats frozen v2 (0.209) and does not raise zh->en translations. Otherwise report the fine-tune as tried and not adopted.
- Then score the chosen system once on ASCEND test and ServiceNow zh.

## 2026-11-14 23:14 — Fine-tune adoption decided on validation
- HF generate() crashed (transformers EncoderDecoderCache bug); run_ft.py now uses its own greedy decoder with a KV cache. Same model.
- Validation MER: auto 0.266, frozen v2 0.209, A (ft alone) 0.155, B (ft preferred in v2) 0.155, C (ft as 4th candidate) 0.166. zh->en translations: auto 31, v2 16, A/B 9.
- Per the rule above, the fine-tuned decode is adopted (A and B tie; B picks ft on 209/210, so the transcript is effectively ft).
- With frozen v2 thresholds, the flag on the ft transcript flags 56–59% of lines at 0.32 precision (the old disagreement threshold no longer fits a much better transcript). So a v3 flag is tuned on VALIDATION ONLY (scripts/tune_v3.py, cands ft > prompt_zh > auto > prompt_en) and frozen to data/thresholds_v3_frozen.json before any test scoring.

## 2026-11-14 23:36 — ASCEND test scored once with frozen v3 (thresholds frozen 23:17:20)
- MER: auto 0.194, prompt_zh 0.185, frozen v2 0.172, v3 (fine-tuned, picks ft on 361/369) 0.120.
- Auto-classified zh->en translations: auto 29, v2 18, v3 14. Auto-classified bad lines: auto 114, v2 95, v3 74.
- v3 flag: 22% of lines, precision 36/83 = 0.43, recall 36/74 = 0.49 (automatic proxy, not hand labels).
- At this point the fine-tuned transcripts had not been hand-labeled (done afterwards, see 00:26).

## 2026-11-15 00:02 — ServiceNow en_es scored once with frozen v2 (Spanish has no fine-tuned model)
- MER: auto 0.064, prompt_es 0.059, prompt_en 0.259, v2 0.059. Plain-Whisper es->en translations 2/259, both flagged and fixed by v2.
- v2 flag on its own transcript: 11 flags (4%), 0 on lines the proxy calls bad (only 2 bad lines). Reported as-is.

## 2026-11-15 00:26 — Blind human labels on the fine-tuned (v3) output
- Set: the same 71 random test lines + the 29 auto-classified translation lines (data/label_set2.jsonl). Plain Whisper (auto) and v3 shuffled together, unnamed; 24 v3 outputs identical in tokens to plain were shown once and the label used for both. 147 labels in data/labels_v3.jsonl.
- Consistency: plain Whisper re-labels agree with the earlier labels on 69/71.
- Random 70 labeled (1 skip): fully correct plain 26 (37%) vs v3 28 (40%); silent translation 8 (11%) vs 7 (10%); dropped 10 vs 6; hallucination 6 vs 6; mishearing 20 vs 23.
- Of 21 lines where plain Whisper translated (earlier labels), v3: 8 still translated, 4 correct, 7 mishearing, 2 hallucination.
- v3 flag vs these labels: random 11/11 flags were real errors, recall 11/42; all 100: 25/28 precision, recall 25/64, 8/11 translations flagged.
- Reading: token-level MER drops 38%, but line-level "fully correct" barely moves; most fixed lines go from badly wrong to slightly wrong.

## 2026-11-15 00:45 — MER recomputed with the standard formula (reporting only)
- An outside review pointed out that reported MER divided each line's edits by the longer of reference and output, then weighted by reference length. That caps the penalty for insertions (hallucinated or looping output).
- Reported MER is now total edits / total reference tokens (scripts/mer_standard.py; score_v2.py and evaluate.py updated). The flag's internal disagreement signal is unchanged, so no transcript or flag changes.
- Same frozen outputs, new MER. ASCEND test: auto 0.233, prompt_zh 0.256, prompt_en 0.372, v1 0.275. Validation: auto 0.267, v2 0.212, ft 0.161, so the adoption decision is the same. ServiceNow zh: auto 0.066, v3 0.077; es: auto 0.066, v2 0.063.

## 2026-11-15 01:00 — Bug fix in the broken-decode check (found on test clip 00606)
- Plain Whisper looped "go go go…" 223 times on 00606. broken() judged every other decode "nearly empty" against that loop, so all decodes counted as broken and v3 kept the loop (flagged).
- Fix: the "nearly empty" comparison is skipped when the auto decode itself looks broken. No thresholds changed.
- Effect, checked on every set: validation 0 lines changed, ServiceNow zh 0, ServiceNow es 0, ASCEND test 1 line (00606, still flagged). 00606 is not in either human-labeled set, so the human-label results are unchanged.
- ASCEND test after the fix: v2 0.175, v3 0.123. v3 vs auto: 10.9 points (bootstrap CI 6.0–19.9), 47% relative (CI 33–62%). Before the fix: v2 0.213, v3 0.161 (31%).

## 2026-11-15 01:02 — v4 plan (written before any new decode exists)
- Scope: the transcript stays the frozen v3 transcript. v4 changes only the flag, so the blind human labels on v3 transcripts (data/labels_v3.jsonl) stay valid as the answer key.
- Disclosure: at ~00:58 an exploratory logistic model (11 features from the saved decodes, fit on validation) was scored against test references to see whether a learned flag had headroom. It ranked lines with 20%+ of tokens wrong at AUC 0.83 on test. That look motivated v4; no v4 feature set, model or threshold was chosen from it, and human labels were not used.
- New decodes (validation first, then test): the fine-tuned decode again with per-token log-probabilities saved (scripts/run_ft.py --tokens; text must match the frozen decode), and Whisper's explicit translate-task output (scripts/run_translate.py).
- v4 flag: one logistic risk score fit on VALIDATION ONLY. Target: the v3 transcript has 20% or more of its reference tokens wrong. Feature groups: (a) the v3 signals (disagreement with each cross-check, mean log-prob, rate, duration, auto-detected language), (b) token confidence (lowest token log-prob, share of low-confidence tokens), (c) translate-match (overlap between the transcript's English words and the translate-task output that the Mandarin-forced decode does not share).
- Selection rule: groups (b) and (c) are each kept only if they raise leave-one-out AUC on validation. The threshold is the validation F1 optimum, the same criterion as v3. Then freeze with a timestamp and score ASCEND test once: against references on all 369 lines and against the human labels.
- Adoption rule: v4 replaces the v3 flag in the write-up only if, on the 70 random human-labeled lines, it flags more real errors than v3 (11) at precision of at least 85%. Otherwise it is reported as tried and not adopted.
- Single-language speech: ASCEND validation and test clips labeled zh or en (at least 1 s; validation 719, test 584) are decoded with plain Whisper and the full v3 system. Validation is read first. If v3 is worse than plain Whisper on validation single-language MER, a gate is designed on validation; otherwise test is scored once as-is.

## 2026-11-15 01:08 — translate-match signal dropped before fitting
- Whisper large-v3-turbo does not translate when given the translate task: on the first 30 validation clips it returned a Mandarin transcript (e.g. "smart phone越来越普及了" -> "Smartphone越来越普及了。"). The signal carries nothing with this model, so feature group (c) is removed and the test set is not decoded this way. The partial validation output was discarded. Groups (a) and (b) proceed as planned.

## 2026-11-15 01:21 — v4 scored once on ASCEND test (model frozen 01:08:46 on validation)
- Validation fit (scripts/tune_v4.py): leave-one-out AUC 0.870 with the v3 signals; adding token-confidence features gave 0.869, so they were not kept. Threshold 0.361 (validation precision 0.75, recall 0.80). Word-mark cut-off: token log-prob below -0.60 (validation: 62% of marked words wrong, 48% of wrong words marked).
- The re-decode with token log-probs reproduced the frozen fine-tuned text on 210/210 validation and 369/369 test clips.
- Test, against references (369 lines, 103 with 20%+ tokens wrong): v3 flag 22% of lines, catches 54/103; v4 flag 29% of lines, catches 65/103. Risk score AUC 0.824.
- Test, human labels, 70 random lines: v3 11/11 flags real errors, 11/42 errors, 4/7 translations. v4 12/15 (80%), 12/42 errors, 6/7 translations. All 99 labeled: v3 25/28, 25/64, 8/11; v4 30/37, 30/64, 10/11.
- **Decision by the rule written at the top of the v4 plan: v4 is NOT adopted** (precision 80% is under the 85% bar). The v3 flag stays the reported flag. v4 is reported as tried.
- Word marks (new output, no earlier version to replace): on test, 196 of 347 marked words were wrong (56%, CI 51-62%) against 529 wrong words among 5,470 (9.7%); 37% of wrong words were marked. Full printout: data/test_v4_report.txt.

## 2026-11-15 08:58 — Single-language speech scored (frozen v3, no changes)
- Validation read first (719 clips): Mandarin 14.7% -> 10.4%, English 27.0% -> 22.1%. v3 is better on both, so by the rule in the v4 plan no gate was designed.
- Test scored once (584 clips): Mandarin (414) 15.2% -> 10.0%, English (170) 26.0% -> 21.3%. All test speech (953 clips, with the 369 mixed): 20.4% -> 12.4% (CI of the difference 5.4-12.3 points). Printout: data/mono_report.txt.
- v3 flag rate: Mandarin 15%, English 44% (75 lines; 60 with at least one error against the reference).
- Intermediate files mono_*_a/_b.jsonl are decode stages; mono_validation.jsonl and mono_test.jsonl are the complete run files.
