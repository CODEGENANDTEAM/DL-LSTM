"""Generate a tiny synthetic MIDI corpus for smoke-testing the pipeline.

The point is to exercise every stage end to end -- parse, augment, encode,
train, sample, decode -- before you download a real dataset. If the model
cannot overfit these (they are mechanically simple and highly repetitive),
something in the plumbing is broken and no amount of real data will fix it.

    python scripts/make_dummy_corpus.py --out data/raw/dummy --n 24
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pretty_midi

# Scale degrees as semitone offsets from the tonic.
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
TRIADS = {"I": [0, 4, 7], "IV": [5, 9, 12], "V": [7, 11, 14], "vi": [9, 12, 16]}
PROGRESSIONS = [["I", "V", "vi", "IV"], ["I", "IV", "V", "I"], ["vi", "IV", "I", "V"]]

# One bar of melody rhythm, in beats. `None` is a rest.
#
# These exist because a corpus with a single note duration has exactly ONE
# inter-onset interval, which makes rhythm_diversity identically 0 no matter
# what the model does -- the metric cannot distinguish a good model from a bad
# one on such data. A smoke corpus should exercise every metric it is scored
# with, so the melody needs several distinct durations and some silence.
RHYTHMS = [
    [1.0, 1.0, 1.0, 1.0],
    [0.5, 0.5, 1.0, 2.0],
    [1.5, 0.5, 1.0, 1.0],
    [0.5, 0.5, 0.5, 0.5, 2.0],
    [2.0, 1.0, 1.0],
    [1.0, None, 1.0, 1.0],          # a rest on beat 2
    [0.5, 0.5, 1.0, None, 1.0],
    [3.0, 1.0],
]


def make_piece(rng: random.Random, tonic: int, bpm: float) -> pretty_midi.PrettyMIDI:
    """One 8-bar piece: a stepwise melody over a repeating chord progression."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0)
    beat = 60.0 / bpm

    scale = rng.choice([MAJOR, MINOR])
    progression = rng.choice(PROGRESSIONS)

    time = 0.0
    degree = 0
    for bar in range(8):
        chord = TRIADS[progression[bar % len(progression)]]
        # Accompaniment: one sustained triad per bar.
        for offset in chord:
            piano.notes.append(
                pretty_midi.Note(
                    velocity=60,
                    pitch=tonic - 12 + offset,
                    start=time,
                    end=time + 4 * beat,
                )
            )
        # Melody: one bar of a varied rhythm, stepwise motion with occasional
        # leaps. Durations sum to 4 beats so the bar grid stays intact.
        offset = 0.0
        for length in rng.choice(RHYTHMS):
            if length is None:                      # a rest: advance, emit nothing
                offset += 1.0
                continue
            degree = max(0, min(len(scale) - 1, degree + rng.choice([-1, -1, 1, 1, 2])))
            octave = 12 * rng.choice([0, 0, 0, 1])
            piano.notes.append(
                pretty_midi.Note(
                    velocity=85,
                    pitch=tonic + scale[degree] + octave,
                    start=time + offset * beat,
                    end=time + (offset + length) * beat,
                )
            )
            offset += length
        time += 4 * beat

    midi.instruments.append(piano)
    return midi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/dummy", help="output directory")
    parser.add_argument("--n", type=int, default=24, help="number of pieces")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        tonic = rng.choice([60, 62, 64, 65, 67])
        midi = make_piece(rng, tonic, bpm=rng.choice([90.0, 100.0, 120.0]))
        midi.write(str(out_dir / f"dummy_{i:03d}.mid"))

    print(f"wrote {args.n} pieces to {out_dir}")


if __name__ == "__main__":
    main()
