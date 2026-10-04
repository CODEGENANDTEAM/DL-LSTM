# Music Generation with LSTMs

Symbolic music generation: MIDI in, MIDI out. The model is a next-token
predictor over a vocabulary of musical symbols -- exactly like character-level
text generation, with notes instead of characters.

```
MIDI corpus -> parse -> quantize -> SPLIT -> augment -> encode -> vocab
            -> sliding windows -> LSTM -> sampled symbols -> MIDI
```

Two swappable seams make the experiments a config change rather than a rewrite:

| Seam | Where | Options |
|---|---|---|
| **Encoding** | `src/data/encode.py` | `note_chord`, `interval`, `pitch_duration`, `event` |
| **Architecture** | `src/models/` | `lstm`, `transformer` |

## Documentation

- [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) -- full walkthrough of the code,
  design decisions, results and likely viva questions.
- [`docs/guide/Code_Guide.pdf`](docs/guide/Code_Guide.pdf) -- 12-page illustrated
  guide split into five presenter sections, plus corrections for the slide deck.
  Source: `docs/guide/code_guide.html` (print to PDF from a browser).

## Results (final model, v5)

| | |
|---|---|
| Data | ADL Piano MIDI, Classical + Jazz + Blues: 1,654 pieces, split 1,324 / 162 / 161, train transposed to 15,745 |
| Vocabulary | 37,020 note/chord symbols (`min_freq: 10`) |
| Model | Embedding 256 -> 3 x LSTM 512 -> Linear 256 + ReLU -> adaptive softmax, 16.9 M params |
| Best val loss | **3.615** (perplexity 37.2), epoch 3 |
| In-key % of samples | **0.884** (real music 0.84-0.88, random notes 0.583) |
| Training time | ~11 min on an RTX 5070 (v1 on CPU took ~8 h) |

| Run | Change | Best val loss |
|---|---|---|
| v1 | CPU, 2 x 256 LSTM, 100-token context, +/-3 keys | 3.685 |
| v2 | GPU, 3 x 512 LSTM, 256-token context | 3.710 |
| v3 | +/-6 transposition | 3.714 |
| v4 | data-pipeline audit fixes, tempo map | 3.682 |
| **v5** | adaptive softmax, plateau LR schedule | **3.615** |

## Setup

Requires Python 3.11+. On the original development machine the `python` on
PATH is a Microsoft Store 3.7 stub and will NOT work -- torch stopped
supporting 3.7 at version 1.13, so `pip install -r requirements.txt` under it
fails with "Could not find a version that satisfies torch>=2.6.0".

Create the virtualenv with a full 3.11+ interpreter (on that machine,
`D:/PYTH/python.exe`), not with the 3.7 `python`:

```bash
D:/PYTH/python.exe -m venv venv
```

Then activate it and install:

```bash
venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

If you already made a venv with the wrong interpreter, `python --version`
inside it will say 3.7.9. Delete the folder and redo the step above -- the
interpreter a venv is built from cannot be changed afterwards.

For GPU training, install the CUDA build of torch instead of the CPU wheel:
https://pytorch.org/get-started/locally/

## Run it: `main.py`

One entry point for the whole pipeline. `main.py` orchestrates the same code
the `scripts/` entry points call, so the two stay equivalent. After setup,
always run through the venv interpreter, `venv\Scripts\python.exe`.

```bash
venv\Scripts\python.exe main.py demo        # zero setup: synthetic corpus, full run, a few seconds
venv\Scripts\python.exe main.py configs     # what experiments are shipped
```

The main experiment end to end (prepare is skipped if cached, then train and
generate), followed by audio rendering and scoring. Pick a new `name` so the
existing v5 checkpoint in `runs/adl_mixed_lstm_v5/` is not overwritten:

```bash
venv\Scripts\python.exe main.py --config configs/adl_mixed.yaml --set name=adl_mixed_lstm_v6
venv\Scripts\python.exe scripts/render_audio.py data/generated
venv\Scripts\python.exe main.py evaluate --config configs/adl_mixed.yaml --set name=adl_mixed_lstm_v6
```

Sample from the already-trained v5 model without retraining:

```bash
venv\Scripts\python.exe main.py generate --config configs/adl_mixed.yaml --num-samples 6
```

| Command | Does |
|---|---|
| `prepare` | raw MIDI -> processed token cache |
| `train` | train the configured model |
| `generate` | sample MIDI (`--temperature`, `--num-samples`, `--seed-midi`) |
| `evaluate` | score `data/generated/` against the reference corpus |
| `all` | prepare (skipped if cached) + train + generate -- the default |
| `demo` | synthetic corpus + the whole pipeline on `configs/smoke.yaml` |
| `configs` | list the configs with their encoding and architecture |

`--config` and `--set key=value` work on either side of the subcommand:

```bash
venv\Scripts\python.exe main.py train --config configs/transformer.yaml --set train.lr=0.0005
```

In `all`/default mode the prepare stage is skipped when the processed cache for
that config already exists; `--force-prepare` rebuilds it.

## Quickstart (synthetic data, ~30 seconds)

The same thing spelled out, one script per stage -- `main.py demo` runs exactly
this.

```bash
venv\Scripts\python.exe scripts/make_dummy_corpus.py --out data/raw/dummy --n 24
venv\Scripts\python.exe scripts/prepare_data.py  --config configs/smoke.yaml
venv\Scripts\python.exe scripts/run_train.py     --config configs/smoke.yaml
venv\Scripts\python.exe scripts/run_generate.py  --config configs/smoke.yaml
```

Output lands in `data/generated/` as `.mid` plus a `.txt` of the raw symbol
sequence for debugging.

## Real data

The main experiment uses the ADL Piano MIDI dataset (11,073 files in 18 genre
folders; `adl-piano-midi.zip`, not committed). Unzip it to `data/raw/adl/` so
each genre is a subfolder; `configs/adl_mixed.yaml` selects Classical, Jazz and
Blues with `data.include_styles`.

Other corpora: drop MIDI into `data/raw/<corpus>/` and point `data.raw_dir` at it.

| Dataset | Size | Notes |
|---|---|---|
| Classical Piano MIDI (piano-midi.de) | ~300 files | Mechanically sequenced, already on a grid. Best starting point. |
| Nottingham | ~1,200 tunes | Tiny vocabulary, trains in minutes. |
| JSB Chorales | 382 pieces | Standard polyphony benchmark. |
| MAESTRO v3 (MIDI-only archive) | ~1,276 performances | Highest quality, but human-performed -- expressive timing needs the quantizer. |

Take the MIDI-only download for MAESTRO; the full release with audio is 100+ GB
and nothing here needs it.

Style-conditioned mixing expects `data/raw/mixed/<style>/*.mid` -- the
subdirectory name becomes the style label. Mixing corpora *without*
`encoding.style_tokens: true` averages incompatible grammars and produces
output that is coherent in none of them.

## Running experiments

Every experiment is a YAML file in `configs/`. Override any field inline:

```bash
venv\Scripts\python.exe scripts/run_train.py --config configs/default.yaml --set train.lr=0.0005
```

Shipped configs: `default` (every setting, commented), `adl_mixed` (the main
v5 experiment), `adl_events` (event encoding), `interval_encoding`,
`transformer`, `mixed_styles`, `smoke`.

Build the comparison table across arms:

```bash
venv\Scripts\python.exe scripts/run_evaluate.py \
  --arm "lstm=data/generated/baseline_lstm_*.mid" \
  --arm "transformer=data/generated/baseline_transformer_*.mid" \
  --out results.md
```

Metrics: perplexity, pitch-class histogram distance, in-key %, note repetition
rate, pitch range, average interval. `in_key_pct` is the one that catches the
interval encoder's characteristic pitch drift; `repetition_rate` catches a
sampler that has collapsed into repeating one note.

## MP3 input (optional)

```bash
pip install basic-pitch
venv\Scripts\python.exe app/transcribe.py song.mp3 --quantize
venv\Scripts\python.exe scripts/run_generate.py --config configs/default.yaml --seed-midi song.mid
```

Transcription is for **inference-time seeding only**. Note accuracy on a full
mix runs roughly 50-70%, and training on that teaches the model transcription
artifacts rather than music. Training data is always clean symbolic MIDI.

## Build order

If something breaks, work down this list -- each step is verified by the one
before it, and steps 1-5 catch nearly every bug that would otherwise surface as
"trained for six hours and the output is noise".

1. **Round-trip a MIDI file** through parse and decode. Listen to it.
2. **Round-trip the symbols** through encode and decode. `pytest tests/` covers
   this for every scheme.
3. **Eyeball one batch** -- check `x[1:]` lines up with `y[0]`.
4. **Overfit one piece deliberately.** Loss should approach zero. If it cannot
   memorise a single file, the plumbing is broken, not the hyperparameters.
5. **Generate from that overfit model** -- it should regurgitate the training piece.
6. Only then train on the full corpus.

## Traps this codebase already handles

- **Split before augmenting.** Transposing first puts transposed copies of
  validation pieces into training; val loss then looks great and means nothing.
  `tests/test_roundtrip.py` asserts this cannot happen.
- **Vocabulary travels with the checkpoint.** `vocab.json` is written beside
  `best.pt`, and generation refuses to run if the checkpoint's data-config hash
  does not match. A mismatched vocabulary maps every token to the wrong note and
  fails silently.
- **Vocabulary built from the training split only.** Building it over val/test
  leaks.
- **Deterministic vocabulary ordering.** Set iteration order is not stable
  across runs; ids come from `sorted()`.
- **Never window across a piece boundary.** The model would learn the end of one
  piece as a valid predecessor of the start of another.
- **Sample, don't argmax.** Greedy decoding collapses into one repeated note.
- **The processed cache is hashed** on `data` + `augment` + `encoding`, so
  changing the grid or the scheme produces a new cache rather than silently
  training on a stale one.

## Layout

```
configs/          one YAML per experiment
src/config.py     config loading + hashing
src/data/
  types.py        NoteEvent, Piece, Encoder protocol  <- the contract
  parse.py        MIDI -> quantized Piece
  augment.py      split (by piece) + transposition
  encode.py       SEAM 1: Piece <-> symbols
  vocab.py        symbol <-> id
  dataset.py      sliding windows, DataLoaders, processed cache
src/models/       SEAM 2: lstm.py, transformer.py
src/sampling.py   temperature / top-k / top-p
src/train.py      training loop, checkpointing, early stopping
src/generate.py   autoregressive sampling
src/decode.py     Piece -> MIDI, duration fill, audio synth (timbre chosen here)
src/evaluate.py   metrics + markdown report
scripts/          CLI entry points (+ render_audio.py, make_dummy_corpus.py)
app/transcribe.py optional MP3 -> MIDI seeding
tests/            383 tests: round-trip, leakage, data audit, efficiency
docs/             CODE_GUIDE.md and the illustrated PDF guide
```

## Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -q
```
