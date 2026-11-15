# Verbatim: catching speech-to-text that silently translates bilingual speakers

**Track:** Languages · **Demo video:** https://youtu.be/Uo6DG6jCHGo · **Demo site:** https://brandonlign.github.io/verbatim/

## The problem
Many bilingual people mix two languages in one sentence. Speech-to-text sometimes doesn't write down what they said. It writes a fluent translation instead. In one real clip, "那你有没有什么favourite foods" ("do you have any favorite foods?") came out of Whisper as **"That you have no favorite food?"**, a different sentence with the opposite meaning, that reads perfectly fine. As transcripts become records (AI-scribe visit notes, voicemails, captions), errors like this go unnoticed. About 27 million people in the US have limited English proficiency ([KFF](https://www.kff.org/racial-equity-and-health-policy/issue-brief/overview-of-health-coverage-and-care-for-individuals-with-limited-english-proficiency/)).

## What we built
**For:** a scribe, clinician or family member reading a transcript of a bilingual speaker, who needs to know which lines to check against the audio.

**In:** audio of mixed Mandarin–English (or Spanish–English) speech. **Out:** a transcript that keeps the speaker's mix of languages, with a flag and reasons on lines to re-check.

1. **Fine-tuned transcriber.** Whisper's decoder, trained on 8,387 real code-switched Mandarin–English clips.
2. **Cross-checks.** Regular Whisper decodes the same clip three ways: auto-detect, Mandarin forced + mixed-language prompt, and English forced + mixed-language prompt.
3. **Choose and flag.** Broken versions are dropped (loops, impossible word rate, wrong script, empty output), and the most confident version is kept. The line is flagged when the model is unsure, when the versions disagree, or when a third language is detected.
4. **Mark the words.** Inside each line, the words the model was least sure of are underlined, so the reader knows where to listen.
5. **Demo:** speak a sentence, or replay a real test clip, and see plain Whisper next to Verbatim (`demo/`). The [hosted copy](https://brandonlign.github.io/verbatim/) shows saved results for the test clips; live recording needs the local server.

![Demo: plain Whisper invents a sentence; Verbatim keeps the Chinese, flags the line and underlines the invented words](docs/demo.png)

*A real test clip. The speaker said "个人的personalities还有那个". Verbatim's line is still wrong, and it says so: the line is flagged and the invented words are underlined.*

## What we added
Our own idea, closest to suggested project 4.4.3. We added our own question (translation instead of transcription), a blind hand-labeled answer key by a native Chinese reader on our team, a fine-tuned fix plus a review flag, and a second unseen dataset with a second language pair.

## How it works
- **Data:** [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND), real Mandarin–English conversation (CC BY-SA 4.0).
  - Train: 8,387 clips.
  - Validation: 210 mixed clips.
  - Test: 369 mixed clips.
  - Speakers are different in each split.
- **Unseen second test:** [ServiceNow asr_codeswitched](https://huggingface.co/datasets/ServiceNow-AI/asr_codeswitched), synthetic speech: 294 Mandarin–English and 259 Spanish–English clips.
- **Model:** [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).
  - Decoder fine-tuned with the encoder frozen: 800 steps × 8 clips, about 2.5 hours on an M3 laptop.
  - Cross-checks run with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
- **Fair testing:** every threshold and model choice was made on validation and saved with a timestamp before the test set was scored once. There were two early looks at test data: a pilot on 80 test clips before the method existed, and one printout of error rates by language group while designing v1. The saved settings are in `data/thresholds_*_frozen.json`.
- **Word marks:** a word is underlined when its token log-probability is below −0.60, a cut-off chosen on validation and saved before the test set was scored.
- **Measures:**
  - Mixed error rate (MER): total edits (substitutions, deletions, insertions) divided by total reference tokens, counting Chinese characters and English words.
  - Blind labels: a native Chinese reader labeled 347 transcripts in two rounds, with the systems shuffled and unnamed. Re-labeling 71 plain-Whisper lines matched the first round on 69.

## Results
All numbers are on the ASCEND test set: two speakers, neither heard in training or validation, scored once after every setting was frozen.

**In short:**
1. **47% fewer word errors** on mixed-language speech: 23.3% → 12.3% (CI 33–62% relative). Both speakers improved.
2. **When Verbatim flags a line, it is wrong:** 11 of 11 flags on a random, blind, human-labeled sample (CI 74–100%). The flag catches about a quarter of the errors.
3. **Underlined words are wrong 56% of the time**, against 10% for words overall, so the reader knows where to listen.

| ASCEND test | Plain Whisper | Verbatim |
|---|---|---|
| **Word errors** (mixed error rate) | | |
| 369 mixed-language clips | 23.3% | **12.3%** |
| All 953 clips, including 584 single-language ones | 20.4% | **12.4%** |
| **Translations** (blind human labels) | | |
| Lines plain Whisper translated, still translated | 21 of 21 | **8 of 21** |
| **Review aids** (Verbatim only) | | |
| Flagged lines that had a real error (70 random lines) | — | **11 of 11** |
| Lines with an error that were flagged (70 random lines) | — | 11 of 42 |
| Underlined words that were wrong (all 369 clips) | — | **196 of 347** |

**Error rate vs. whole lines.** On 70 random lines judged blind, 40% of Verbatim's lines were fully correct against 37% for plain Whisper. Most lines it fixes go from badly wrong to slightly wrong, which the error rate counts and a line-level label does not. That is why the flag and underlines are part of the system.

**Single-language speech improves too:** Mandarin-only 15.2% → 10.0% (414 clips), English-only 26.0% → 21.3% (170 clips).

**Second, unseen dataset** (ServiceNow, synthetic speech): Spanish–English 6.6% → 6.3%, and both of plain Whisper's Spanish→English translations were fixed without any Spanish fine-tuning. Mandarin–English went 6.6% → 7.7%, from writing-style differences (below). Plain Whisper never translated this clean synthetic Mandarin (0 of 294) but did on 29 of 369 real clips by the automatic count, so synthetic benchmarks miss the problem.

**Where it fails:**
- **Short phrases at a language switch** still get translated: "就是hard choice" → "this is hard choice".
- **When every decode makes the same mistake,** nothing is flagged: "使用这个function的时候" → "使用这个方向的时候".
- **Writing style carries over from ASCEND:** numbers come out as Chinese characters ("ID七四零五六一七"), which costs points on other datasets.
- **The flag doesn't transfer yet:** it fires on 44% of English-only lines (60 of 75 had an error) and found no real errors in Spanish (0 of 11).

Full tables, the v4 flag we tried and did not adopt, and examples: [RESULTS.md](RESULTS.md).

## What we'd do next
1. Fine-tune on a mix of real code-switched sources, including Spanish–English conversation, and train harder on short switch-point phrases.
2. Test with real clinic conversations from bilingual patients, with their permission.

## Sources
- **Datasets:**
  - ASCEND, Lovenia et al. 2022 ([paper](https://arxiv.org/abs/2112.06223), [data](https://huggingface.co/datasets/CAiRE/ASCEND)).
  - ServiceNow-AI [asr_codeswitched](https://huggingface.co/datasets/ServiceNow-AI/asr_codeswitched).
- **Models and libraries:**
  - OpenAI [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) and its [mlx-community conversion](https://huggingface.co/mlx-community/whisper-large-v3-turbo).
  - [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
  - Hugging Face [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org).
  - [OpenCC](https://github.com/yichen0831/opencc-python), pandas, soundfile, soxr.
- **Prior work:**
  - ServiceNow AI, [code-switching ASR benchmark](https://huggingface.co/blog/ServiceNow-AI/code-switching) (2026).
  - Peng et al. 2023, [prompting Whisper](https://arxiv.org/abs/2305.11095).
  - Whisper on ASCEND, [arXiv 2510.18724](https://arxiv.org/abs/2510.18724).
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic) helped write code and analysis.

## Run it
Needs an Apple Silicon Mac on macOS 14 or later, Python 3.12 and FFmpeg (for microphone audio). [uv](https://docs.astral.sh/uv/) or plain `venv` both work.

```bash
git clone https://github.com/brandonlign/verbatim && cd verbatim
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_verbatim.py                 # 12 tests, recomputes the headline numbers from saved outputs
.venv/bin/python scripts/get_weights.py                 # downloads the fine-tuned decoder (688 MB) and rebuilds the model
.venv/bin/python demo/server.py --ft models/ft_decoder  # then open http://localhost:8780
```

- **Weights:** only the decoder was trained, so the [release](https://github.com/brandonlign/verbatim/releases/tag/v1.0) holds just the 100 changed tensors. `get_weights.py` loads the public base model, swaps them in and checks all 587 tensors against `models_sha256.json`. Training from scratch instead: `scripts/finetune.py`, about 2.5 hours on an M3.
- **Without `--ft`,** the demo runs the regular-Whisper version (v2).
- **Every experiment:** [RESULTS.md#reproduce](RESULTS.md#reproduce).

The result files in `data/` include reference transcripts from ASCEND (CC BY-SA 4.0, shared under the same license) and ServiceNow asr_codeswitched. Seven short ASCEND test clips are included in `demo/audio/` for the demo (same license). The fine-tuned weights are a derivative of Whisper (MIT) trained on ASCEND (CC BY-SA 4.0).
