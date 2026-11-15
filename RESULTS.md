# Verbatim: full results

The timeline of every decision, with timestamps, is in [NOTES_INTEGRITY.md](NOTES_INTEGRITY.md).

One disclosure: before any method existed, a first pilot ran on 80 clips that are inside the ASCEND test set, and that pilot shaped the starting idea (forcing the language plus a mixed-language prompt). Every threshold and model choice after that was made on validation.

## Versions
| Version | Transcript | Flag thresholds frozen |
|---|---|---|
| v1 | Mandarin forced + prompt, falls back to auto-detect if it comes back nearly empty | 2026-11-14 18:50 (validation) |
| v2 | Most confident of 3 regular Whisper decodes, broken ones dropped | 2026-11-14 20:19 (validation) |
| **v3 (final)** | **Fine-tuned decoder preferred, regular decodes as cross-checks** | **2026-11-14 23:17 (validation)** |

Fine-tuning was adopted by a rule written before any fine-tuned output was seen: adopt it only if it beats v2 on validation MER and doesn't raise translations. Validation MER: plain 26.7%, v2 21.2%, fine-tuned 16.1%.

MER is total edits (substitutions, deletions, insertions) divided by total reference tokens (Chinese characters plus English words, traditional characters converted to simplified). An earlier draft normalized each line by the longer of reference and output, which under-counts insertions; every MER here uses the standard formula on the same frozen outputs (logged 2026-11-15 00:45).

## ASCEND test (369 mixed-language clips, held out)
| System | MER | Chinese→English translations (automatic count) | Lines the automatic classifier calls bad |
|---|---|---|---|
| Plain Whisper (auto-detect) | 23.3% | 29 | 114 |
| Mandarin forced + prompt | 25.6% | 5 | 108 |
| English forced + prompt | 37.2% | 23 | 112 |
| v1 | 27.5% | 7 | — |
| v2 | 21.3% | 18 | 95 |
| **v3 (fine-tuned)** | **16.1%** | **14** | **74** |
| Fine-tuned decode alone (not the chosen system) | 12.3% | 13 | 74 |

- v3 vs plain: the MER difference is 7.2 points (bootstrap 95% CI 5.6–8.8), a 31% relative cut (CI 21–42%).
- Per line: v3 was better on 165, worse on 36 and the same on 168.
- v3 picked the fine-tuned decode on 361 of 369 clips. Most of the gap to the fine-tuned decode alone is one clip where plain Whisper looped ("go go go…" 223 times): the broken-decode check marked the fine-tuned decode "nearly empty" next to the loop, every decode counted as broken, and v3 kept the loop. The line was flagged. The check is left as frozen.
- The automatic count over-counts translations. On the 29 lines it marked, a native reader confirmed 14 as translations and 12 as hallucinations. Human labels are the answer key.

### Blind human labels
**Round 1:** plain Whisper vs v1, 200 transcripts.

**Round 2:** plain Whisper vs v3, 147 transcripts.

Both rounds covered the same 71 random test lines plus 29 lines the automatic count had marked as translations. Re-labeled plain-Whisper lines agreed with round 1 on 69 of 71.

| 70 random lines (round 2) | Plain Whisper | v3 |
|---|---|---|
| Fully correct | 26 (37%, CI 27–49%) | 28 (40%, CI 29–52%) |
| Silent translation | 8 (11%) | 7 (10%) |
| Dropped words | 10 | 6 |
| Hallucination | 6 | 6 |
| Mishearing | 20 | 23 |

- **Plain Whisper's 21 human-confirmed translations under v3:** 8 still translated, 4 fully correct, 7 mishearing, 2 hallucination.
- **v3 flag, the 70 random lines:** 11 of 11 flags were real errors (CI 74–100%). It caught 11 of 42 errors (26%, CI 15–41%) and 4 of 7 translations.
- **v3 flag, all 99 labeled lines** (the 70 random plus 29 picked because the automatic count marked them as translations, so errors are over-represented): 25 of 28 flags were real errors (89%, CI 73–96%). It caught 25 of 64 errors (39%) and 8 of 11 translations.
- On all 369 test lines, v3 flags 22%.
- **v1 flag, round 1 (on v1's transcript):** 42 of 50 flags were real errors (84%, CI 71–92%), and it caught 17 of 21 translations (81%, CI 60–92%). v1's transcript wasn't better than plain Whisper: 4 translations vs 7, but 10 hallucinations vs 6.

## ServiceNow asr_codeswitched (synthetic speech, never used for tuning)
| Language pair | System | MER | English-word recall | Other-language-word recall | Translations into English |
|---|---|---|---|---|---|
| Mandarin–English (294) | Plain Whisper | 6.6% | 0.72 | 0.96 | 0 |
| | v3 | 7.7% | 0.62 | 0.96 | 1 |
| Spanish–English (259) | Plain Whisper | 6.6% | 0.88 | 0.99 | 2 |
| | v2 (no fine-tune) | 6.3% | 0.89 | 1.00 | 0 |

- **Mandarin:** v3 is worse on this set because of writing conventions it learned from ASCEND: numbers as Chinese characters ("ID七四零五六一七" for "ID7405617") and spelled-out acronyms ("M C C").
- **Spanish:** v2 caught and fixed both translations. One was "I sent a note para mover el ID463938 a otro support team…", which plain Whisper rendered as "I sent a note to move the ID 463938 to another support team…". The Spanish flag raised 11 lines (4%), and none were real errors. One English→Spanish translation went unflagged in every decode ("FollowMe is installed, but it doesn't show up in the printer list" → "…no aparece en la lista de las libras").

## Examples (ASCEND test)
| Said | Plain Whisper | v3 |
|---|---|---|
| 那你有没有什么favourite foods | That you have no favorite food? | 那你有没有什么favorite food |
| 哦这么technical | Oh, so technical. | 哦这么technical的 |
| 然后当了one semester | And then one semester. | 然后当了one semester |
| not what I expected in the beginning but just对最后就回到这里 | It's not what I expected in the beginning, but just, right, I'll just go back to this. | it's not what i expected in the beginning but just对最后就回到这里 |
| 就是hard choice (failure) | — | this is hard choice |
| 个人的personalities还有那个 (failure, flagged) | — | 个人的personalities oh I want to go |

## Reproduce
These steps need an Apple Silicon Mac. Data comes from Hugging Face (`CAiRE/ASCEND`, `ServiceNow-AI/asr_codeswitched`) and is placed in `data/` as parquet files.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# regular Whisper decodes
.venv/bin/python scripts/run_asr.py data/ascend_test.parquet data/asr_test.jsonl --only-mixed
.venv/bin/python scripts/add_decode.py data/ascend_test.parquet data/asr_test.jsonl data/asr_test3.jsonl
# fine-tune and decode
.venv/bin/python scripts/finetune.py data/ascend_train models/ft_decoder --max-steps 800 --bs 8 --lr 1e-5
.venv/bin/python scripts/run_ft.py models/ft_decoder data/ascend_test.parquet data/asr_test3.jsonl data/asr_test_ft.jsonl
# tune on validation (same steps on the validation split first), then score test once
.venv/bin/python scripts/mer_standard.py   # every MER in this file
.venv/bin/python scripts/tune_v3.py data/asr_val_ft.jsonl data/thresholds_v3_frozen.json
.venv/bin/python scripts/score_v2.py data/asr_test_ft.jsonl data/thresholds_v3_frozen.json zh --extra ft --cands ft,prompt_zh,auto,prompt_en
# ServiceNow
.venv/bin/python scripts/run_sn.py data/servicenow/en_es.parquet data/sn_es.jsonl es
.venv/bin/python scripts/score_v2.py data/sn_es.jsonl data/thresholds_v2_frozen.json es
# demo at http://localhost:8780
.venv/bin/python demo/server.py --ft models/ft_decoder
```

**Files:**
- `scripts/verbatim.py`: choose and flag.
- `scripts/taxonomy.py`: the automatic error classifier.
- `scripts/label_server.py`: the blind labeling tool.
- `data/labels_test.jsonl` and `data/labels_v3.jsonl`: human labels.
- `data/thresholds_*_frozen.json`: frozen settings.
