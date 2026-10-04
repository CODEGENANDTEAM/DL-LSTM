"""Optional audio -> MIDI frontend, for seeding generation only.

INFERENCE-TIME SEEDING ONLY. Never build training data with this.

Spotify's basic-pitch transcribes a full mix at roughly 50-70% note accuracy:
it misses inner voices, invents octave errors, smears sustained notes and adds
spurious onsets. A model trained on that corpus learns the transcriber's
artifacts -- not music. The training source is clean symbolic MIDI, always.
Humming a phrase and asking the model to continue it is a legitimate use;
transcribing an album to enlarge the corpus is not.

basic-pitch is a soft dependency: it is imported inside ``transcribe`` so the
rest of the project runs without it. Install with ``pip install basic-pitch``.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Union

import pretty_midi

__all__ = ["transcribe", "quantize_midi", "transcribed_seed"]

_INSTALL_HINT = (
    "audio seeding needs Spotify's basic-pitch, which is not installed.\n"
    "    pip install basic-pitch\n"
    "It is intentionally optional: transcription is only ever used to seed "
    "generation, never to build training data."
)


def transcribe(audio_path: Union[str, Path], out_midi_path: Union[str, Path]) -> Path:
    """Transcribe an audio file (mp3/wav/...) to MIDI at ``out_midi_path``.

    The result is a rough note transcription suitable only as a generation
    seed. Returns the written MIDI path.
    """
    try:
        # Lazy: keeps basic-pitch (and its tensorflow/onnx stack) off the
        # critical path for everyone who only trains on symbolic MIDI.
        from basic_pitch.inference import predict_and_save
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_INSTALL_HINT) from exc

    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"audio file not found: {audio}")
    out = Path(out_midi_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # basic-pitch chooses its own output filename inside a directory, so run it
    # in a scratch dir and move the single .mid it produces into place.
    # ignore_cleanup_errors: on Windows a handle still open in the TF/ONNX
    # stack makes rmtree raise, which would turn a successful transcription
    # into an exception after the MIDI was already written.
    with tempfile.TemporaryDirectory(prefix="basic_pitch_", ignore_cleanup_errors=True) as scratch:
        predict_and_save(
            audio_path_list=[str(audio)],
            output_directory=scratch,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
        )
        produced = sorted(Path(scratch).glob("*.mid*"))
        if not produced:
            raise RuntimeError(f"basic-pitch produced no MIDI for {audio}")
        shutil.move(str(produced[0]), str(out))

    return out


@contextlib.contextmanager
def transcribed_seed(audio_path: Union[str, Path], grid: float) -> Iterator[Path]:
    """Transcribe + quantize ``audio_path`` into a temp dir that is removed on exit.

    Yields the quantized MIDI path; parse it inside the ``with`` block::

        with transcribed_seed(audio, grid=float(cfg.data.grid)) as midi_path:
            piece = load_piece(midi_path, cfg)

    src/generate.py's ``_transcribe_seed`` used ``tempfile.mkdtemp()`` and
    returned a path inside it, so every ``--seed-audio`` run leaked a directory
    holding the raw and the quantized transcription. This owns the lifetime.
    """
    with tempfile.TemporaryDirectory(prefix="seed_transcribe_", ignore_cleanup_errors=True) as tmp:
        audio = Path(audio_path)
        raw = transcribe(audio, Path(tmp) / f"{audio.stem}.mid")
        yield quantize_midi(raw, grid=grid)


def quantize_midi(
    midi_path: Union[str, Path],
    grid: float = 0.25,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Snap note onsets and durations of a MIDI file onto a beat grid.

    ``grid`` is in quarter-note beats and should come from ``cfg.data.grid`` so
    a seed lands on the same lattice the model was trained on. Raw
    transcription output is wildly off-grid, and an off-grid seed encodes to
    symbols the vocabulary has never seen.

    Returns the written path (``<stem>.quantized.mid`` by default).
    """
    src = Path(midi_path)
    dst = Path(out_path) if out_path else src.with_suffix(".quantized.mid")
    if grid <= 0:
        raise ValueError(f"grid must be positive, got {grid}")

    midi = pretty_midi.PrettyMIDI(str(src))
    _, tempi = midi.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0
    if tempo <= 0:
        tempo = 120.0
    seconds_per_beat = 60.0 / tempo

    for instrument in midi.instruments:
        snapped = []
        for note in instrument.notes:
            start_beats = round((note.start / seconds_per_beat) / grid) * grid
            dur_beats = round((note.get_duration() / seconds_per_beat) / grid) * grid
            dur_beats = max(grid, dur_beats)  # never quantize a note out of existence
            note.start = max(0.0, start_beats) * seconds_per_beat
            note.end = note.start + dur_beats * seconds_per_beat
            snapped.append(note)
        instrument.notes = snapped

    dst.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(dst))
    return dst


def main(argv: Optional[list[str]] = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_config

    parser = argparse.ArgumentParser(
        description="Transcribe audio to MIDI for use as a generation seed."
    )
    parser.add_argument("audio", help="input audio file (mp3, wav, ...)")
    parser.add_argument("-o", "--out", default=None, help="output .mid path")
    parser.add_argument("--config", default=None, help="config used for --quantize grid")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="snap the transcription onto cfg.data.grid (recommended for seeds)",
    )
    args = parser.parse_args(argv)

    audio = Path(args.audio)
    out = Path(args.out) if args.out else audio.with_suffix(".mid")

    midi_path = transcribe(audio, out)
    print(f"[transcribe] wrote {midi_path}")

    if args.quantize:
        cfg = load_config(args.config)
        midi_path = quantize_midi(midi_path, grid=float(cfg.data.grid), out_path=out)
        print(f"[transcribe] quantized to a {cfg.data.grid}-beat grid: {midi_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
