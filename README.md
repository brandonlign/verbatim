# Verbatim: speech-to-text that writes down what bilingual people actually say

**Track:** Languages · **Demo video:** https://youtu.be/Uo6DG6jCHGo · **Try the demo:** https://brandonlign.github.io/verbatim/

![Demo: plain Whisper invents a sentence; Verbatim keeps the Chinese, flags the line and underlines the invented words](docs/demo.png)

## The problem
A lot of bilingual people switch languages in the middle of a sentence. When Whisper hears a sentence like that, it often doesn't transcribe it. It quietly writes an English translation of what it thinks you meant, and sometimes that translation is wrong.

One speaker in our test set asked "那你有没有什么favourite foods" ("do you have any favorite foods?"). Whisper wrote **"That you have no favorite food?"** That's a different sentence with the opposite meaning, and it reads perfectly fine. Nobody looking at the transcript would know anything happened unless they went back to the audio.

That matters more every year, because transcripts are turning into records: AI-scribe visit notes, voicemails, captions. About 27 million people in the US have limited English proficiency ([KFF](https://www.kff.org/racial-equity-and-health-policy/issue-brief/overview-of-health-coverage-and-care-for-individuals-with-limited-english-proficiency/)), and many of them mix languages when they talk. When a transcript translates them, words end up on the record that they never said.

## What we built
Verbatim takes audio of mixed Mandarin–English (or Spanish–English) speech and gives back a transcript that keeps the speaker's own mix of languages. It also tells the reader which lines to double-check, and which words in those lines to listen to again. It's built for whoever reads the transcript: a scribe, a clinician, a family member.

It has three parts:

- **A fine-tuned Whisper.** We trained the decoder of Whisper large-v3-turbo on 8,387 real code-switched Mandarin–English clips, so it learns that mixing languages is normal and should be written down as spoken. Training took about 2.5 hours on an M3 MacBook.
- **Cross-checks and a flag.** The same clip also runs through regular Whisper three ways: auto-detect, Mandarin forced with a mixed-language prompt, and English forced with a mixed-language prompt. Broken outputs (loops, impossible word rates, wrong script, empty text) are thrown out, and the most confident remaining transcript is kept. That's almost always the fine-tuned one. If the model is unsure, the versions disagree, or a third language shows up, the line is flagged.
- **Underlined words.** Inside each line, the words the model was least sure about are underlined, so the reader knows exactly where to listen.

The demo shows plain Whisper and Verbatim side by side. You can speak a mixed sentence into it or replay real test clips. The [hosted version](https://brandonlign.github.io/verbatim/) shows saved results for the test clips. Live recording needs the local server (see [Run it](#run-it)).

## Results
Our main test is the ASCEND test set: 369 clips of real mixed Mandarin–English conversation from two speakers the model never heard in training or tuning. Every setting was frozen before we scored it, and we scored it once.

**47% fewer word errors.** The mixed error rate fell from 23.3% to 12.3% (95% CI for the cut: 33–62%). Both test speakers improved a lot, one from 34.1% to 17.2% and the other from 17.4% to 9.7%. Verbatim was better on 166 lines and worse on 36.

**It stops translating.** A native Chinese reader on our team confirmed 21 lines where plain Whisper translated instead of transcribing. Verbatim kept the speaker's own words on 13 of them.

**When it flags a line, trust the flag.** On 70 random lines labeled blind, all 11 lines Verbatim flagged had a real error (95% CI 74–100%). Across all 99 labeled lines, 25 of 28 flags were real. It flagged 11 of the 42 lines with an error in the random sample. It's built to be precise, so a reader's time goes to lines that really need a second listen.

**The underlines point straight at the mistakes.** An underlined word is wrong 56% of the time. Across all words, that number is 10%. An underlined word is almost six times as likely to be wrong.

**Normal speech gets better too.** Fine-tuning on mixed speech didn't hurt single-language speech. It improved it: Mandarin-only clips went from 15.2% to 10.0% errors, and English-only from 26.0% to 21.3%. Across all 953 test clips, errors fell from 20.4% to 12.4%.

**It fixed Spanish without any Spanish training.** On ServiceNow's synthetic Spanish–English set, which we never used for tuning, the cross-check system caught and fixed both translations plain Whisper made, and errors went from 6.6% to 6.3%.

| ASCEND test | Plain Whisper | Verbatim |
|---|---|---|
| Word errors on 369 mixed-language clips | 23.3% | **12.3%** |
| Word errors on all 953 clips (mixed + single-language) | 20.4% | **12.4%** |
| Human-confirmed translations that stayed translated | 21 of 21 | **8 of 21** |
| Flagged lines that really had an error (70 random lines) | — | **11 of 11** |
| Underlined words that were really wrong | — | **196 of 347 (56%)** |

**Synthetic benchmarks miss this problem.** On ServiceNow's clean synthetic Mandarin–English speech, plain Whisper translated 0 of 294 clips. On real conversation it did it over and over. If you only test on synthetic speech, you won't see the problem at all.

**What the error rate means for a whole line.** Most of what fine-tuning fixes takes a line from badly wrong to nearly right, often one character off. On the 70 random lines, 40% of Verbatim's lines were word-for-word perfect against 37% for plain Whisper. That's why the flag and the underlines are part of the system: they point the reader to the mistakes that are left.

## How we kept the results honest
- **Separate speakers everywhere.** ASCEND train (8,387 clips), validation (210 mixed clips) and test (369 mixed clips) have no speakers in common.
- **Frozen before testing.** Every threshold and model choice was made on validation and saved with a timestamp before test was scored. The files are in `data/thresholds_*_frozen.json`, and a test checks the timestamps.
- **Rules written in advance.** We wrote down when we would adopt the fine-tuned model before we saw any of its output. We also built a smarter learned flag (v4). It missed the precision bar we set beforehand, so we didn't use it, and it's reported in [RESULTS.md](RESULTS.md).
- **Blind human labels.** A native Chinese reader labeled 347 transcripts in two rounds, with the systems shuffled and unnamed. Re-labeling 71 plain-Whisper lines matched the first round on 69.
- **Early looks, disclosed.** Before the method existed, a pilot ran on 80 test clips, and we printed error rates by language group once while designing v1. Everything after that was tuned on validation. The full timeline is in [NOTES_INTEGRITY.md](NOTES_INTEGRITY.md).

## Where it still falls short
- **Short phrases right at a language switch** can still get translated: "就是hard choice" became "this is hard choice".
- **If every decode makes the same mistake, there's nothing to disagree about,** so the flag stays quiet: "使用这个function的时候" became "使用这个方向的时候".
- **It picked up ASCEND's writing style.** It writes numbers as Chinese characters ("ID七四零五六一七"), which costs points on other datasets. On ServiceNow's Mandarin set, errors went from 6.6% to 7.7%.
- **The flag is tuned for mixed Mandarin–English.** It fires on 44% of English-only lines (60 of those 75 did have an error), and in Spanish it raised 11 flags with no real errors behind them.

## What's next
1. Train on a mix of real code-switched sources, including Spanish–English conversation, with extra focus on short phrases at the switch point.
2. Test on real clinic conversations with bilingual patients, with their permission.

## Under the hood
- **Data:** [ASCEND](https://huggingface.co/datasets/CAiRE/ASCEND), real Mandarin–English conversation (CC BY-SA 4.0), and [ServiceNow asr_codeswitched](https://huggingface.co/datasets/ServiceNow-AI/asr_codeswitched), synthetic speech: 294 Mandarin–English and 259 Spanish–English clips.
- **Model:** [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo). Decoder fine-tuned with the encoder frozen, 800 steps of 8 clips. Cross-checks run with [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
- **Underlines:** a word is underlined when its token log-probability is below −0.60, a cut-off chosen on validation.
- **Mixed error rate (MER):** total edits (substitutions, deletions, insertions) divided by total reference tokens, counting Chinese characters and English words.

Every table, the v4 flag, more examples and the commands to reproduce each number: [RESULTS.md](RESULTS.md).

## Where this fits
This started as our own idea, closest to suggested project 4.4.3. What we added: our own question (translation instead of transcription), a blind hand-labeled answer key, a fine-tuned model with a review flag and word underlines, and a second unseen dataset with a second language pair.

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
You'll need an Apple Silicon Mac on macOS 14 or later, Python 3.12 and FFmpeg (for microphone audio). [uv](https://docs.astral.sh/uv/) or plain `venv` both work.

```bash
git clone https://github.com/brandonlign/verbatim && cd verbatim
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_verbatim.py                 # 12 tests, recomputes the headline numbers from saved outputs
.venv/bin/python scripts/get_weights.py                 # downloads the fine-tuned decoder (688 MB) and rebuilds the model
.venv/bin/python demo/server.py --ft models/ft_decoder  # then open http://localhost:8780
```

- **Weights:** only the decoder was trained, so the [release](https://github.com/brandonlign/verbatim/releases/tag/v1.0) holds just the 100 changed tensors. `get_weights.py` loads the public base model, swaps them in and checks all 587 tensors against `models_sha256.json`. To train from scratch instead, run `scripts/finetune.py` (about 2.5 hours on an M3).
- **Without `--ft`,** the demo runs the regular-Whisper version (v2).
- **Every experiment:** [RESULTS.md#reproduce](RESULTS.md#reproduce).

The result files in `data/` include reference transcripts from ASCEND (CC BY-SA 4.0, shared under the same license) and ServiceNow asr_codeswitched. Seven short ASCEND test clips are included in `demo/audio/` for the demo (same license). The fine-tuned weights are a derivative of Whisper (MIT) trained on ASCEND (CC BY-SA 4.0).
