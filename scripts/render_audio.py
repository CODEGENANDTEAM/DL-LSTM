"""Render generated MIDI to WAV so you can actually listen to it.

A .mid file contains no sound -- only instructions. Something has to synthesize
it. This renders to 44.1 kHz stereo WAV in pure Python (numpy + the stdlib
`wave` module, see src/decode.py), so every machine can play the result.

The default `piano` voice is an additive model of a struck string: percussive
attack, pitch-dependent two-stage decay, stretched partials, detuned unison
strings, velocity-dependent brightness, a small synthetic room, and loudness
normalised to about -14 dBFS RMS with a peak limiter. It is recognisably a
piano but not a sampled one; for a realistic instrument pass a General MIDI
soundfont with --soundfont (needs FluidSynth), or open the .mid in MuseScore.

    python scripts/render_audio.py                      # everything in data/generated
    python scripts/render_audio.py path/to/file.mid
    python scripts/render_audio.py --instrument organ --out-dir data/generated/organ_compare
    python scripts/render_audio.py --soundfont path/to/GeneralUser.sf2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.parse import MIDI_SUFFIXES  # noqa: E402
from src.decode import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TARGET_RMS_DB,
    INSTRUMENTS,
    RenderStats,
    render_wav,
)

SAMPLE_RATE = DEFAULT_SAMPLE_RATE


def render(midi_path: Path, out_path: Path, fs: int = SAMPLE_RATE, **options) -> Path:
    """Synthesize one MIDI file to a 16-bit stereo WAV. Returns the written path.

    ``options`` are passed to ``src.decode.synthesize_midi`` (instrument,
    reverb, velocity_shaping, seed, target_rms_db, soundfont).
    """
    return render_wav(midi_path, out_path, fs=fs, **options).path


def midi_files_in(directory: Path) -> List[Path]:
    """The MIDI files directly inside ``directory``, in a platform-independent order.

    Only the top level: rglob would descend into e.g. data/generated/organ_compare.
    ``glob("*.mid")`` matched ``.MID`` on Windows but not on Linux, never
    matched ``.midi``, and sorted case-sensitively on Linux only.
    """
    return sorted(
        (f for f in directory.iterdir() if f.suffix.lower() in MIDI_SUFFIXES and f.is_file()),
        key=lambda f: (f.name.lower(), f.name),
    )


def _fmt(stats: RenderStats) -> str:
    return (f"{stats.seconds:5.1f}s audio  RMS {stats.rms_db:6.1f} dBFS "
            f"(whole-file {stats.plain_rms_db:6.1f})  peak {stats.peak_db:5.1f} dBFS  "
            f"rendered in {stats.render_seconds:.2f}s")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=None,
                        help="MIDI files or directories (default: data/generated)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write WAVs (default: alongside each MIDI)")
    parser.add_argument("--rate", type=int, default=SAMPLE_RATE, help="sample rate (Hz)")
    parser.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="piano",
                        help="built-in voice; 'organ' is the old renderer's sound")
    parser.add_argument("--target-rms", type=float, default=DEFAULT_TARGET_RMS_DB,
                        help="loudness target, gated RMS in dBFS (default %(default)s)")
    parser.add_argument("--reverb", type=float, default=None,
                        help="room wet mix 0-1 (default: instrument preset, ~0.2)")
    parser.add_argument("--no-reverb", action="store_true", help="dry output")
    parser.add_argument("--velocity-shaping", choices=["auto", "on", "off"], default="auto",
                        help="add accents/voicing dynamics; 'auto' does it only when "
                             "all velocities are equal (model output)")
    parser.add_argument("--seed", type=int, default=0, help="seed for velocity jitter")
    parser.add_argument("--soundfont", default=None,
                        help="render with FluidSynth and this .sf2 instead of the built-in synth")
    args = parser.parse_args(argv)

    targets: List[Path] = []
    for raw in (args.paths or ["data/generated"]):
        p = Path(raw)
        targets.extend(midi_files_in(p) if p.is_dir() else [p])

    if not targets:
        print("no MIDI files found. Generate some first:\n"
              "    python main.py demo", file=sys.stderr)
        return 2

    options = dict(
        instrument=args.instrument,
        reverb=0.0 if args.no_reverb else args.reverb,
        velocity_shaping=args.velocity_shaping,
        seed=args.seed,
        target_rms_db=args.target_rms,
        soundfont=args.soundfont,
    )
    out_dir = Path(args.out_dir) if args.out_dir else None
    ok = 0
    for midi_path in targets:
        dest = (out_dir / midi_path.name if out_dir else midi_path).with_suffix(".wav")
        try:
            stats = render_wav(midi_path, dest, fs=args.rate, **options)
        except RuntimeError as exc:  # e.g. FluidSynth missing: same for every file
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            print(f"  {midi_path.name}: FAILED ({exc})", file=sys.stderr)
            continue
        ok += 1
        note = "" if stats.notes else "  (no notes: wrote silence)"
        print(f"  {dest}\n      {_fmt(stats)}{note}")

    print(f"\nrendered {ok}/{len(targets)} file(s). Double-click a .wav to play it.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
