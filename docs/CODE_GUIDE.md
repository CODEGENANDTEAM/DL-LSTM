# Code Guide — Symbolic Music Generation with LSTMs

A complete walkthrough of this codebase for someone who has never opened it.
Read it top to bottom and you should be able to answer any question about what
the code does, why it is built this way, and what the results mean.

An illustrated version split into five presenter sections, with diagrams, is in
[`guide/Code_Guide.pdf`](guide/Code_Guide.pdf) (source: `guide/code_guide.html`).

---

## 1. What the project does, in one sentence

It learns to compose piano music by predicting the next musical symbol, the same
way a language model predicts the next word — except the "words" are notes and
chords, and the "text" is MIDI files.

That one idea drives everything else:

| Language model | This project |
|---|---|
| Text file | MIDI file |
| Word / token | A note (`60`) or a chord (`60.64.67`) |
| Vocabulary | 37,020 distinct note and chord symbols |
| "Predict the next word" | "Predict the next musical event" |
| Generated sentence | Generated music, written back out as MIDI |

The model is an **LSTM**, a recurrent neural network. A Transformer is also
implemented, as a comparison.

---

## 2. Running it

```
venv\Scripts\python.exe main.py demo                              # ~5 s, synthetic data, proves the install works
venv\Scripts\python.exe main.py --config configs/adl_mixed.yaml --set name=adl_mixed_lstm_v6   # the real thing: prepare + train + generate (new name keeps v5's checkpoint)
venv\Scripts\python.exe main.py generate --config configs/adl_mixed.yaml --num-samples 6
venv\Scripts\python.exe scripts/render_audio.py data/generated    # MIDI to playable WAV
venv\Scripts\python.exe -m pytest tests/ -q                       # 383 tests
```

`main.py` subcommands: `prepare`, `train`, `generate`, `evaluate`, `all`
(the default), `demo`, `configs`. Any setting can be overridden inline with
`--set section.key=value`.

**Important:** use `venv\Scripts\python.exe`. The `python` on PATH is a Python
3.7 stub that cannot install modern PyTorch.

---

## 3. The pipeline, end to end

```
 MIDI files                         data/raw/adl/<Genre>/.../*.mid
     |   parse.py       read, quantize to a 16th-note grid, filter
     v
 Piece objects          lists of NoteEvent(pitch, start, duration, velocity)
     |   augment.py     SPLIT into train/val/test, then transpose TRAIN only
     v
 More Pieces            1,324 train pieces -> 15,745 transposed copies
     |   encode.py      Piece -> text symbols  ["60", "60.64.67", "<REST:3>"]
     v
 Symbol sequences
     |   vocab.py       symbol <-> integer id, built from TRAIN only
     v
 Integer sequences      12.8 M training tokens, cached as .npy
     |   dataset.py     cut into overlapping 256-token windows
     v
 (x, y) tensors         x = 256 ids, y = the same shifted one step left
     |   train.py       the LSTM learns to predict y from x
     v
 best.pt + vocab.json   the trained model, in runs/<name>/
     |   generate.py    prime with real music, then sample new tokens
     v
 New symbols
     |   encode.py      decode symbols back into notes
     |   decode.py      write MIDI, then synthesize audio
     v
 .mid + .wav            data/generated/
```

### Stage 1 — Parsing (`src/data/parse.py`)

Reads MIDI with `pretty_midi` and produces `Piece` objects.

- **Quantization.** Every note's start and length is snapped to a 0.25-beat grid
  (a 16th note). Real performances have times like 0.4271 beats; unsnapped,
  every duration would be unique and the vocabulary would explode.
- **Time is in beats, not seconds.** Beats are musical; seconds are not.
- **`tempo_map: true`** maps beats through *every* tempo change. 16% of files
  change tempo mid-piece, and using only the first tempo pushed 263 files off
  their notated beats, the worst by more than 600 beats.
- **Filters.** Drops notes shorter than `min_duration`, pieces with fewer than
  `min_events` notes, and pieces whose time signature is not in the allow-list.
- **`track_strategy: merge`** combines all non-drum tracks. That is safe here
  because every ADL track is a piano voice. For a full band arrangement, use
  `densest`, or bass, guitar and melody get flattened into one line.
- **Duplicate notes** at the same onset are merged. They used to produce fake
  chords like `60.60`.
- **Style label** comes from the top-level folder: `Classical`, `Jazz`, `Blues`.

### Stage 2 — Split and augment (`src/data/augment.py`)

**Order matters: split first, augment second.** Transposing first would place
transposed copies of validation pieces into training, and validation would then
be measuring memorisation. A test enforces the order.

- `split_pieces` splits by *piece*, with a fixed seed, and **drops held-out
  pieces that duplicate a training piece**, even in a different key. ADL
  contains the same piece under different filenames; 7 held-out pieces were
  exact copies of training pieces.
- `transpose_augment` shifts every training piece into all 13 keys from -6 to +6
  semitones, so 1,324 pieces become 15,745.

### Stage 3 — Encoding (`src/data/encode.py`) — **seam 1**

Turns a `Piece` into text symbols. Four interchangeable schemes:

| Scheme | A symbol looks like | Vocabulary | Notes |
|---|---|---|---|
| **`note_chord`** (in use) | `60`, `60.64.67` | 37,020 | A whole chord is one symbol |
| `interval` | `+2`, `-5` | ~400 | The arXiv "Differential Music" idea |
| `pitch_duration` | `P60`, `D1.000` | small | Melody only, no chords |
| `event` | `N60`, `D4`, `T2` | 184 | MIDI-like; learns real durations |

**The time model.** Exactly one symbol per grid step advances the clock by one
step. Silence uses run-length rests, so `<REST:3>` means three silent steps.
Before run-length rests, 56% of all tokens were `<REST>`; afterwards, 36%.

**Why `include_duration: false`.** Putting note length inside the chord symbol
multiplies the vocabulary by every duration variant — 363,455 symbols when
tried. Length is reconstructed at generation time instead (stage 9).

### Stage 4 — Vocabulary (`src/data/vocab.py`)

Maps symbols to integers, built from the **training split only**; using val or
test would leak.

- Reserved ids first: `<PAD>`=0, `<UNK>`=1, `<BOS>`=2, `<EOS>`=3, `<REST>`=4.
- Then corpus symbols in `sorted()` order — **never** set order, which varies
  between runs and would make a reloaded checkpoint decode every token wrongly.
- `min_freq: 10` drops symbols seen fewer than 10 times. 60% of raw symbols
  appear exactly once and can never be learned, yet each costs an embedding row
  and a softmax column. Pruning sends about 4% of tokens to `<UNK>` and shrinks
  the vocabulary roughly 14×.
- Symbols an encoder *declares* (rest runs, style tokens) are exempt from
  pruning. Losing a rest loses **time**, which shifts every later note.

### Stage 5 — Windows and cache (`src/data/dataset.py`)

- Cuts each sequence into windows of `seq_len` (256) tokens, stepping by
  `window_stride` (128), so each token is seen at two context positions.
- **Windows never cross a piece boundary**, or the model would learn that one
  piece's ending leads into another's beginning.
- **Teacher forcing:** `y` is `x` shifted one token left, so a 256-token window
  gives 256 training targets rather than one.
- One extra end-aligned window per piece ensures piece endings are trained on.
- Results are cached as `.npy` in `data/processed/<hash>/`, where the hash
  covers the `data`, `augment` and `encoding` sections. Change the grid or the
  scheme and you get a *new* cache instead of silently reusing a stale one.

### Stage 6 — The model (`src/models/lstm.py`) — **seam 2**

```
Embedding(37,020 -> 256)          9.5 M params    one learned vector per symbol
      |
LSTM layer 1  (256 -> 512)        1.6 M           recurrent, 4 gates
LSTM layer 2  (512 -> 512)        2.1 M
LSTM layer 3  (512 -> 512)        2.1 M
      |
Dropout(0.3) -> Linear(512 -> 256) -> ReLU
      |
Adaptive softmax over 37,020 symbols
                                 -------
                                 16.9 M trainable parameters
```

- Each LSTM layer is a standard cell with **forget, input, cell-candidate and
  output gates**. Initialisation uses orthogonal recurrent weights, which stop
  the repeated matrix product from exploding across 256 steps, and a forget-gate
  bias of 1, so the cell starts out remembering.
- **Output shape is `[batch, seq_len, vocab]`** — a prediction at every
  position. Generation slices the last one.
- **Adaptive softmax** puts frequent symbols in a small "head" group and rare
  ones in narrower clusters. Probabilities stay exact, it is 41% faster per
  batch, and VRAM drops from 5.9 GB to 1.2 GB. `model.output: full` restores the
  plain softmax.
- `src/models/transformer.py` is the comparison arm: identical interface,
  8-head self-attention with a causal mask.

### Stage 7 — Training (`src/train.py`)

- **Loss:** cross-entropy over every position, ignoring `<PAD>`.
- **Optimizer:** Adam, learning rate 0.001, gradients clipped at 5.0.
- **Backpropagation through time** across all 256 timesteps.
- **Early stopping** with `patience: 3`. In every run here, once validation loss
  was stale for two epochs it never recovered.
- **`lr_schedule: plateau`** halves the learning rate the first epoch validation
  stops improving.
- **Checkpoints** are written atomically (temp file, then rename) so an
  interrupted save cannot corrupt `best.pt`, and `vocab.json` is always saved
  next to the model. A checkpoint without its vocabulary is useless.
- **Resume** (`train.resume: true`) restores the weights *and* Adam's state.
- **A VRAM warning** fires before training if the settings look too big for the
  card, because on Windows an overflow silently spills into system RAM and
  epochs become 20–40× slower with no error message.

### Stage 8 — Generation (`src/generate.py`, `src/sampling.py`)

1. Load `best.pt` and the `vocab.json` beside it, and refuse to run if the
   checkpoint and the dataset were built with different vocabularies.
2. **Prime** the model with 256 tokens of real music from the **validation**
   split (`generate.seed_split: val`). Held-out music means a continuation
   cannot simply be recited from memory. You can prime from your own file with
   `--seed-midi`, or from an MP3 with `--seed-audio`.
3. **Sample** the next token from the model's probability distribution.
   Sampling, not argmax: always taking the likeliest token collapses into one
   note repeated forever.
   - `temperature` (0.9) flattens or sharpens the distribution.
   - `top_k` / `top_p` optionally cut off the unlikely tail.
   - Special tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<STYLE:...>`, `<UNK>`) are
     **masked out**. `<UNK>` used to be sampled about 4% of the time and decoded
     as a silent hole in the music.
4. Feed the token back in and repeat, carrying the LSTM's hidden state forward.
   That is 26–34× faster than re-reading the whole window every step.
5. Decode the symbols back into notes and write MIDI.

Every run prints its seed, so any sample can be reproduced with `--seed N`.

### Stage 9 — Audio (`src/decode.py`, `scripts/render_audio.py`)

- `fill_durations`: because `include_duration` is off, every decoded note is one
  grid step long. This stretches each note to the next onset, recovering 80% of
  real note lengths to within half a beat, against 66% for a flat 16th.
  **Without it, every piece sounds like staccato blips.**
- The renderer is a pure-numpy synthesizer: a struck-string envelope, up to 40
  stretched partials, velocity shaping, polyphony-aware mixing, a small stereo
  reverb, normalisation to -14 dBFS and a limiter. Output is 44.1 kHz stereo.
- Options: `--instrument piano|epiano|organ`, and `--soundfont x.sf2` for a real
  sampled instrument if FluidSynth is installed.
- **Timbre is chosen here, at render time.** Nothing the model learned decides
  whether the output sounds like a piano or an organ.

### Stage 10 — Evaluation (`src/evaluate.py`)

| Metric | Meaning | Real music | This model |
|---|---|---|---|
| `perplexity` | How surprised the model is; roughly "how many symbols it is choosing between". Lower is better | — | 37 |
| `in_key_pct` | Fraction of notes inside the estimated key. **The headline metric** | 0.84–0.88 | **0.884** |
| `distinct_n` | Variety of note n-grams | 0.61 | 0.74 |
| `rhythm_div` | Variety of note spacings | 0.394 | 0.363 |
| `repetition_rate` | Fraction of immediately repeated notes | 0.028 | 0.055 |
| `copy_metrics` | Overlap with training data (a plagiarism check) | — | lower than real unseen music |

Random notes score **0.583** in-key, so that is the floor, not zero.

---

## 4. Where everything lives

| File | Responsibility |
|---|---|
| `main.py` | The entry point and CLI; runs the stages in order |
| `src/config.py` | Loads YAML configs, applies `--set` overrides, hashes configs, estimates VRAM |
| `src/data/types.py` | `NoteEvent`, `Piece`, the `Encoder` interface — **the contract every module agrees on** |
| `src/data/parse.py` | MIDI to `Piece`: quantize, filter, tempo map |
| `src/data/augment.py` | Split by piece, drop duplicates, transpose |
| `src/data/encode.py` | Seam 1: `Piece` to symbols and back; four schemes |
| `src/data/vocab.py` | Symbol to id mapping, pruning, save/load |
| `src/data/dataset.py` | Windows, batches, the processed cache |
| `src/models/lstm.py` | Seam 2: the LSTM |
| `src/models/transformer.py` | Seam 2: the Transformer comparison |
| `src/train.py` | Training loop, early stopping, checkpoints, resume |
| `src/generate.py` | Priming, sampling loop, writing output |
| `src/sampling.py` | Temperature, top-k, top-p, token masking |
| `src/decode.py` | `Piece` to MIDI, duration fill, the audio synthesizer |
| `src/evaluate.py` | All metrics and the results table |
| `scripts/` | One script per stage, plus `render_audio.py` and `make_dummy_corpus.py` |
| `app/transcribe.py` | Optional MP3 to MIDI, for priming from audio |
| `tests/` | 383 tests across 7 files |

**The two seams** are the reason experiments are cheap: any encoder and any
architecture plug into the same pipeline, so comparisons are a config change,
not a rewrite.

---

## 5. Configuration

One YAML file describes an experiment. `configs/default.yaml` holds every
setting with a comment explaining it; other configs override parts of it.

| Config | What it is |
|---|---|
| `adl_mixed.yaml` | **The main experiment** (v5): ADL Classical+Jazz+Blues, note_chord, adaptive softmax |
| `adl_events.yaml` | The event-encoding experiment |
| `smoke.yaml` | Tiny synthetic run used by `main.py demo` |
| `transformer.yaml`, `interval_encoding.yaml`, `mixed_styles.yaml` | Comparison arms |

Settings worth knowing:

| Setting | Value here | Why |
|---|---|---|
| `data.grid` | 0.25 | 16th-note quantization |
| `data.tempo_map` | true | Follow tempo changes |
| `augment.transpose_range` | [-6, 6] | All 13 keys |
| `encoding.scheme` | note_chord | Seam 1 |
| `encoding.min_freq` | 10 | Prune the long tail |
| `model.seq_len` | 256 | About 93 beats, roughly 23 bars of memory |
| `model.window_stride` | 128 | Half-window overlap |
| `model.output` | adaptive | Faster, less memory |
| `train.batch_size` | 48 | Fits the 12 GB card |
| `train.patience` | 3 | Stop early; it never recovers |
| `generate.temperature` | 0.9 | Measured closest to real music |
| `generate.seed_split` | val | Prime from held-out music |

---

## 6. Results

| Run | What changed | Best val loss | Perplexity |
|---|---|---|---|
| v1 | First run, CPU, 2×256 LSTM, 100-token context, ±3 keys | 3.685 | 39.9 |
| v2 | Bigger model, 256-token context | 3.710 | 40.8 |
| v3 | Transposition widened to ±6 | 3.714 | 41.0 |
| v4 | Data-pipeline bugs fixed, tempo map | 3.682 | 39.7 |
| **v5** | **Adaptive softmax, LR schedule** | **3.615** | **37.2** |
| events | Event encoding (not comparable) | 1.677 | 5.4 |

Final sample scores: in-key **0.884** against 0.84–0.88 for real music and 0.583
for random notes.

Training v5 takes **11 minutes** on an RTX 5070, down from about 8 hours on CPU
at the start of the project.

---

## 7. Design decisions, and the evidence for them

Every one of these was measured, and several reversed an earlier assumption.

| Decision | Why |
|---|---|
| **Teacher forcing** | Scoring only the last position wasted 99% of each window. Targets per epoch went from 623k to 10.3M |
| **Window stride 128** | Stride 1 gave 9.9M near-identical windows and 50 hours per epoch |
| **Run-length rests** | Rests fell from 56% of tokens to 36%; sequences shortened 26% |
| **Prune below 10 occurrences** | 60% of symbols appeared once. Vocabulary shrank 14× for ~4% `<UNK>` |
| **No `include_duration`** | It would multiply the vocabulary to 363,455 |
| **Duration fill at render time** | Recovers 80% of real note lengths; without it everything is staccato |
| **bf16 mixed precision — rejected** | 25% faster, but validation loss 5.35 versus 4.80. **Speed alone is not enough evidence** |
| **Adaptive softmax — adopted** | 41% faster, VRAM 5.9 GB to 1.2 GB, probabilities still exact |
| **Temperature 0.9, no truncation** | Truncation doubled the repeated-note rate; the model is tonal enough without it |
| **Event encoding — built, not adopted** | Fixed the vocabulary, `<UNK>` and durations, but in-key fell to 0.61, near random |
| **±6 transposition** | No measurable gain over ±3 in either loss or samples. Kept; honest negative result |

---

## 8. Known limitations

- **No long-range form.** A 256-token window is about 23 bars, so the model
  cannot restate a theme from a minute earlier. This is the LSTM's core limit
  and the reason the Transformer arm exists.
- **It overfits after about 3–5 epochs** (v5's best is epoch 3). More varied data would help more than a
  bigger model.
- **Durations are estimated, not learned**, in the note_chord scheme.
- **Style conditioning is weak.** The style token appears only at position 0 of
  a piece, so most training windows never see it.
- **The Transformer arm has not been trained on the full corpus**, only
  smoke-tested. Do not report Transformer results without running it.
- **`<UNK>` covers about 4% of tokens.** Those chords are unrecoverable.
- **The synthesized audio is not a real piano.** Open the MIDI in MuseScore for
  a realistic sound.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| An epoch suddenly takes 10× longer | The GPU is full and Windows is spilling to system RAM. Close games and browsers, or lower `train.batch_size` |
| Training dies overnight with no error | The PC slept. Set sleep to Never, and resume with `--set train.resume=true` |
| `Could not find a version that satisfies torch` | You are using the Python 3.7 stub. Use `venv\Scripts\python.exe` |
| Generation refuses with a vocabulary mismatch | The dataset was rebuilt after the model was trained. Retrain, or restore the old cache |
| Samples sound atonal and aimless | Check `in_key_pct`. Near 0.58 means random; check the model trained, and the sampling temperature |
| Output sounds like staccato blips | `fill_durations` is not being applied |
| The evaluator reports "no MIDI matched" | An arm label contained `=`, or the time-signature filter dropped the generated files |

---

## 10. Likely questions, with answers

**Why an LSTM and not a plain RNN?**
A plain RNN's memory decays within a few steps because gradients vanish. The
LSTM's gates give it a cell state it can carry unchanged, so a motif from 100
steps back can still influence the next note.

**What is teacher forcing?**
During training the model is fed the *true* previous notes rather than its own
predictions, so every position in a window becomes a training target and the
whole window can be scored in one pass.

**What is perplexity?**
The exponential of the loss. Roughly the number of symbols the model is
effectively choosing between. Ours is 37 out of a vocabulary of 37,020, so the
model has narrowed the field by about a factor of a thousand.

**Why sample instead of taking the most likely note?**
Greedy decoding collapses into one note repeated forever, because the most
likely continuation of a repeated note is that same note again.

**How do you know it isn't copying the training data?**
`copy_metrics` compares generated output against the true continuation and
against every training n-gram. The longest verbatim match is about 11 tokens,
and its overlap with the training set is *lower* than real unseen music's.
Generation is also primed with held-out validation music.

**Why is the vocabulary so large?**
Every distinct combination of simultaneous notes is its own symbol. Real piano
music contains tens of thousands of distinct chords.

**What would you do with more time?**
Train the Transformer arm on the full corpus, since long-range structure is the
clearest weakness; and pursue the event encoding, which solves the vocabulary
and duration problems but needs a stronger model to hold a key.

**Which metric matters most, and what's a good value?**
`in_key_pct`. Random notes score 0.583, real music 0.84–0.88, and this model
0.884. Read it alongside `distinct_n`, because a model that repeats one safe
phrase also scores high on key.

**How is the train/test split kept honest?**
Split by piece before augmentation, with duplicate pieces removed from the
held-out sets even when transposed, and the vocabulary built from training data
only.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Token / symbol | One musical event: a note, a chord, or a rest |
| Vocabulary | The set of all symbols the model knows |
| Quantization | Snapping times to a musical grid |
| Transposition | Shifting a whole piece up or down in pitch |
| Teacher forcing | Training on the true previous tokens |
| Perplexity | `exp(loss)`; effective number of choices |
| Temperature | How adventurous sampling is |
| top-k / top-p | Cutting off unlikely options before sampling |
| Epoch | One pass over the training data |
| Early stopping | Halting when validation stops improving |
| Checkpoint | Saved model weights (`best.pt`) |
| Seam | A deliberate swap point: encoding, and architecture |
