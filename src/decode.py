"""Rendering a Piece into a Standard MIDI File, and MIDI into audio.

Two render-time stages live here, and neither looks at vocabulary symbols:

1. **Piece -> MIDI** (``piece_to_midi``, ``write_midi``, ``fill_durations``).
   Turning model symbols back into a Piece is the encoder's job
   (``Encoder.decode``). NoteEvent times are quarter-note beats (see
   src/data/types.py); MIDI wants seconds, so every time is multiplied by
   ``60 / tempo`` on the way out.

2. **MIDI -> audio** (``synthesize_midi``, ``render_wav``). A small additive
   synthesiser in pure numpy -- no soundfont or binary needed -- so every
   machine can listen to the output. See the "Audio rendering" section below.

Nothing in stage 2 changes which notes are played: it only decides how they
sound (timbre, loudness, room), exactly like choosing a GM program.
"""

from __future__ import annotations

import math
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pretty_midi

from src.data.types import NoteEvent, Piece

__all__ = [
    "PROGRAMS",
    "piece_to_midi",
    "write_midi",
    "fill_durations",
    "INSTRUMENTS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TARGET_RMS_DB",
    "partial_spec",
    "shape_velocities",
    "synthesize_midi",
    "loudness_stats",
    "render_wav",
    "RenderStats",
]

# TIMBRE IS CHOSEN HERE, AT RENDER TIME. It has nothing to do with what the
# model learned: the network only ever emits pitches (and durations), never
# instrument identity. Rendering the same Piece with program 0 and program 48
# gives a piano and a string section playing exactly the same notes.
PROGRAMS: Dict[str, int] = {
    "acoustic_grand": 0,
    "electric_piano": 4,
    "harpsichord": 6,
    "guitar": 24,
    "strings": 48,
}

# pretty_midi rejects notes whose end is not strictly after their start.
_MIN_SECONDS = 1e-3


def _resolve_program(program: Union[int, str]) -> int:
    """Accept either a raw GM program number or a PROGRAMS preset name."""
    if isinstance(program, str):
        try:
            return PROGRAMS[program]
        except KeyError as exc:
            raise ValueError(
                f"unknown program preset {program!r}; "
                f"choose one of {sorted(PROGRAMS)} or pass a GM number 0-127"
            ) from exc
    return max(0, min(127, int(program)))


def piece_to_midi(piece: Piece, program: Union[int, str] = 0) -> pretty_midi.PrettyMIDI:
    """Build an in-memory PrettyMIDI object from a Piece.

    Args:
        piece: Source piece; ``piece.tempo`` sets both the written tempo and
            the beats-to-seconds conversion.
        program: General MIDI instrument number (0-127) or a PROGRAMS key.

    Returns:
        A single-instrument PrettyMIDI object.
    """
    # A non-positive tempo would make every note collapse to zero length or
    # run backwards, so fall back to the Piece default rather than emit junk.
    tempo = float(piece.tempo) if piece.tempo and piece.tempo > 0 else 120.0
    seconds_per_beat = 60.0 / tempo

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=_resolve_program(program))

    for event in piece.events:
        # A sampling model can emit out-of-range values -- the interval encoder
        # in particular integrates deltas and drifts off the keyboard -- so
        # clamp instead of letting pretty_midi raise mid-render.
        pitch = max(0, min(127, int(event.pitch)))
        velocity = max(1, min(127, int(event.velocity)))

        start = max(0.0, float(event.start)) * seconds_per_beat
        end = start + max(0.0, float(event.duration)) * seconds_per_beat
        if end - start < _MIN_SECONDS:
            end = start + _MIN_SECONDS

        instrument.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        )

    midi.instruments.append(instrument)
    return midi


def write_midi(
    piece: Piece, path: Union[str, Path], program: Union[int, str] = 0
) -> Path:
    """Render ``piece`` and write it to ``path``. Returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    piece_to_midi(piece, program=program).write(str(out))
    return out


def fill_durations(piece: Piece, max_beats: float = 4.0) -> Piece:
    """Let each note ring until the next onset, capped at ``max_beats``.

    For encodings that carry no durations (``encoding.include_duration: false``)
    the decoder has to invent one, and its default is a single grid step -- a
    16th note. That turns every held chord and sustained melody into a staccato
    blip; round-tripping Satie's Gymnopedie No. 2 through it took the median
    note from 1.25 beats to 0.25, and it is the main reason model output sounded
    random rather than musical.

    Measured over 40 real Classical/Jazz/Blues pieces, against their true
    durations:

        rule                          within 0.5 beat   median length (true 0.50)
        one grid step (old default)        66%               0.25
        until next onset  (this)           80%               0.75
        until next note in register        75%               1.00
        until same pitch re-struck         30%               4.00

    It runs slightly long, which reads as legato -- much closer to real piano
    than staccato. Like the instrument program, this is a render-time choice: it
    changes nothing the model learned, so it is safe on any existing checkpoint.
    """
    events = sorted(piece.events, key=lambda e: (e.start, e.pitch))
    onsets = sorted({e.start for e in events})
    following = dict(zip(onsets, onsets[1:]))
    filled = [
        NoteEvent(
            pitch=e.pitch,
            start=e.start,
            duration=max(e.duration, min(max_beats, following.get(e.start, e.start + max_beats) - e.start)),
            velocity=e.velocity,
        )
        for e in events
    ]
    return Piece(events=filled, source=piece.source, style=piece.style, tempo=piece.tempo)


# ---------------------------------------------------------------------------
# Audio rendering
# ---------------------------------------------------------------------------
#
# Why this exists: the first renderer used pretty_midi.synthesize with a
# 4-harmonic sine stack, one fixed 1-second decay for every pitch, linear
# velocity and peak normalisation. It sounded like a dull organ and, because
# the densest chord set the gain, 4-9 dB quieter than normal music.
#
# The synth below is additive: each note is a sum of decaying partials.
#
#   * Envelope: a few-ms raised-cosine attack (no click), then a two-stage
#     exponential decay like a struck string (a fast "prompt" sound and a slow
#     "aftersound"), shorter for higher pitches, then a damper release.
#   * Timbre: up to 40 stretched (inharmonic) partials, a hammer-position comb,
#     velocity-dependent brightness, higher partials decaying faster, and two
#     slightly detuned "strings" per note that beat against each other.
#     Partials at or above 0.95 * Nyquist are never generated (no aliasing).
#   * Mixing: per-note gain from velocity and n^-0.4 of the n notes sounding
#     at its onset; equal-power stereo pan by pitch.
#   * Room: a synthetic stereo impulse response (band-wise decaying noise)
#     applied by FFT convolution.
#   * Loudness: gated RMS normalised to a target (default -14 dBFS), then a
#     look-ahead peak limiter so nothing clips.
#
# Speed: each note is Im(sum_k a_k z_k^t) with the block factorisation
# z^(bB + j) = z^(bB) * z^j, which turns a whole note into one BLAS matrix
# product. There are no per-sample Python loops.

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_TARGET_RMS_DB = -14.0
_CEILING_DB = -1.0            # limiter ceiling, dBFS (sample peak)
_NYQUIST_MARGIN = 0.95        # never synthesise partials above this * fs/2
_BLOCK = 1024                 # block size for the note factorisation
_WAV_CACHE_BYTES = 256 << 20  # bound on cached per-note waveforms


@dataclass(frozen=True)
class _Preset:
    attack: Tuple[float, float]   # attack seconds at (ff, pp)
    release: float                # damper release time constant, seconds
    min_sound: float              # a struck note sounds at least this long
    hammer: float                 # level of the hammer-noise transient
    reverb: float                 # default wet mix
    pan_width: float              # max |pan| for the extreme registers
    poly_exponent: float          # per-note gain = n_sounding ** -poly_exponent


INSTRUMENTS: Dict[str, _Preset] = {
    "piano": _Preset(attack=(0.0015, 0.006), release=0.08, min_sound=0.05,
                     hammer=0.035, reverb=0.20, pan_width=0.35, poly_exponent=0.4),
    "epiano": _Preset(attack=(0.002, 0.006), release=0.10, min_sound=0.05,
                      hammer=0.0, reverb=0.18, pan_width=0.25, poly_exponent=0.4),
    # The original sound (pretty_midi.synthesize + 4-harmonic stack), kept for
    # A/B comparison. It still goes through the new loudness stage so the
    # comparison isolates timbre rather than volume.
    "organ": _Preset(attack=(0.0, 0.0), release=0.0, min_sound=0.0,
                     hammer=0.0, reverb=0.0, pan_width=0.0, poly_exponent=0.0),
}


def _hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def partial_spec(
    pitch: int, velocity: int = 100, instrument: str = "piano", fs: int = DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs_hz, amplitudes, decay_rates_per_s)`` for one note.

    Every returned frequency is strictly below ``0.95 * fs / 2``. Amplitudes
    are normalised to unit energy so every register starts equally loud
    before velocity is applied.
    """
    f0 = _hz(pitch)
    vn = max(1, min(127, int(velocity))) / 127.0
    limit = _NYQUIST_MARGIN * fs / 2.0

    if instrument == "piano":
        n = np.arange(1, 41, dtype=np.float64)
        B = 1e-4 * 10.0 ** ((pitch - 21) / 40.0)                 # string stiffness
        fn = n * f0 * np.sqrt(1.0 + B * n * n)                    # stretched partials
        amps = n ** -(1.45 - 0.85 * vn)                           # harder = brighter
        amps = amps * (0.35 + 0.65 * np.abs(np.sin(np.pi * n / 7.3)))  # strike point
        amps = amps / (1.0 + (fn / (2500.0 + 6000.0 * vn)) ** 2)  # felt hammer
        # The soundboard radiates little below ~100 Hz (real bass notes have
        # weak fundamentals) and laptop speakers reproduce none of it; energy
        # there only muddies the mix and eats headroom.
        amps = amps / np.sqrt(1.0 + (110.0 / fn) ** 4)
        scale = 2.0 ** (-(pitch - 60) / 14.0)                     # higher = shorter
        r_fast = (1.0 / (0.45 * scale)) * (1.0 + 0.30 * (n - 1))  # prompt sound
        r_slow = (1.0 / (3.0 * scale)) * (1.0 + 0.45 * (n - 1))   # aftersound
        detune = 2.0 ** (0.7 / 1200.0) - 1.0                      # ~0.7 cent unison
        freqs = np.concatenate([fn * (1 + detune), fn * (1 - detune)])
        amps = np.concatenate([0.72 * amps, 0.28 * amps])
        rates = np.concatenate([r_fast, r_slow])
    elif instrument == "epiano":
        n = np.arange(1, 6, dtype=np.float64)
        harm = np.array([1.0, 0.30, 0.10, 0.05, 0.03])
        harm[1:] *= 0.4 + 0.8 * vn
        tau = 1.6 * 2.0 ** (-(pitch - 60) / 18.0)
        freqs = np.concatenate([n * f0, [7.2 * f0]])              # + inharmonic "tine"
        amps = np.concatenate([harm, [0.18 * vn]])
        rates = np.concatenate([(1.0 / tau) * (1.0 + 0.8 * (n - 1)), [1.0 / 0.04]])
    elif instrument == "organ":
        freqs = np.arange(1, 5, dtype=np.float64) * f0
        amps = np.array([1.0, 0.5, 0.25, 0.12])
        rates = np.ones(4)                                         # pretty_midi's 1 s decay
    else:
        raise ValueError(f"unknown instrument {instrument!r}; choose from {sorted(INSTRUMENTS)}")

    keep = (freqs < limit) & (amps > 1e-3 * amps.max())
    freqs, amps, rates = freqs[keep], amps[keep], rates[keep]
    if amps.size:
        amps = amps / math.sqrt(float(np.sum(amps * amps)))
    return freqs, amps, rates


def _decaying_partials(freqs, amps, rates, n: int, fs: int) -> np.ndarray:
    """sum_k amps_k * exp(-rates_k t) * sin(2 pi freqs_k t) for samples 0..n-1.

    Sample b*B + j equals S[b] * P[j], so the note is one complex matmul.
    """
    if n <= 0 or amps.size == 0:
        return np.zeros(max(n, 0))
    s = (2j * np.pi * freqs - rates) / fs                                # (K,)
    # Real string partials are not phase-locked; aligned phases make every
    # attack a needle-sharp spike (~3-6 dB higher crest) for no audible gain.
    # A fixed golden-ratio sequence keeps it deterministic.
    phase = np.exp(2j * np.pi * ((np.arange(amps.size) * 0.6180339887) % 1.0))
    blocks = -(-n // _BLOCK)
    P = np.exp(np.arange(_BLOCK)[:, None] * s[None, :])                   # (B, K)
    S = np.exp((np.arange(blocks) * _BLOCK)[:, None] * s[None, :]) * (amps * phase)
    return (S @ P.T).imag.ravel()[:n]


def shape_velocities(
    starts: np.ndarray, pitches: np.ndarray, velocities: np.ndarray,
    midi: Optional[pretty_midi.PrettyMIDI] = None, seed: int = 0,
) -> np.ndarray:
    """Add deterministic musical dynamics to flat velocities.

    Downbeats and beats are accented and off-beats softened; in a chord the
    top voice (usually the melody) is lifted, the bass slightly, inner voices
    softened; plus a small seeded jitter. Notes must be sorted by (start, pitch).
    """
    v = velocities.astype(np.float64).copy()
    if v.size == 0:
        return velocities.copy()
    base = max(1.0, float(np.median(v))) / 100.0   # offsets are "per velocity 100"

    if midi is not None:
        try:
            beats = np.asarray(midi.get_beats(), dtype=np.float64)
            downbeats = np.asarray(midi.get_downbeats(), dtype=np.float64)
        except Exception:  # noqa: BLE001 - odd tempo maps: skip metric accents
            beats = downbeats = np.zeros(0)

        def near(grid: np.ndarray) -> np.ndarray:
            if grid.size == 0:
                return np.zeros(starts.size, dtype=bool)
            idx = np.clip(np.searchsorted(grid, starts), 0, grid.size - 1)
            prev = np.maximum(idx - 1, 0)
            dist = np.minimum(np.abs(grid[idx] - starts), np.abs(grid[prev] - starts))
            return dist < 0.03

        on_down, on_beat = near(downbeats), near(beats)
        v += base * np.where(on_down, 8.0, np.where(on_beat, 3.0, -4.0))

    # chord roles: notes whose onsets are within 20 ms form one chord
    group = np.concatenate([[0], np.cumsum(np.diff(starts) > 0.02)])
    bounds = np.flatnonzero(np.diff(np.concatenate([[-1], group, [group[-1] + 1]])))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        size = hi - lo
        if size < 2:
            continue
        order = lo + np.argsort(pitches[lo:hi], kind="stable")
        v[order[-1]] += base * (6.0 if size >= 3 else 4.0)
        if size >= 3:
            v[order[0]] += base * 2.0
            v[order[1:-1]] -= base * 8.0

    rng = np.random.default_rng(seed)
    v += base * np.clip(rng.normal(0.0, 3.0, v.size), -6.0, 6.0)
    return np.clip(np.rint(v), 1, 127).astype(np.int64)


def _fast_len(n: int) -> int:
    """Smallest 2^a * 3^b * 5^c >= n (a fast FFT size)."""
    best = 1 << max(0, (n - 1).bit_length())
    f5 = 1
    while f5 < best:
        f35 = f5
        while f35 < best:
            m = f35
            while m < n:
                m *= 2
            best = min(best, m)
            f35 *= 3
        f5 *= 5
    return best


_IR_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _room_ir(fs: int, seed: int = 7) -> np.ndarray:
    """Stereo (2, L) synthetic small-hall impulse response with unit energy.

    Band-limited noise with its own decay per band (highs die first), a
    12 ms pre-delay and a few early reflections. Deterministic.
    """
    key = (fs, seed)
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    rng = np.random.default_rng(seed)
    length = int(1.8 * fs)
    t = np.arange(length) / fs
    nfft = _fast_len(length)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    bands = [(120.0, 500.0, 1.5), (500.0, 2000.0, 1.2),
             (2000.0, 6000.0, 0.8), (6000.0, 20000.0, 0.4)]   # (lo Hz, hi Hz, RT60 s)
    ir = np.zeros((2, length))
    for ch in range(2):
        spec = np.fft.rfft(rng.standard_normal(nfft))
        for lo, hi, rt in bands:
            f = np.maximum(freqs, 1.0)
            mask = 1.0 / (1.0 + (lo / f) ** 4) / (1.0 + (f / hi) ** 4)
            ir[ch] += np.fft.irfft(spec * mask, nfft)[:length] * np.exp(-6.91 * t / rt)
        top = float(np.abs(ir[ch]).max())
        for delay_ms, gain in ((11 + 3 * ch, 0.5), (19 - 2 * ch, 0.35), (29 + 4 * ch, 0.25), (41, 0.18)):
            ir[ch, int(delay_ms * fs / 1000)] += gain * top
    ir = np.concatenate([np.zeros((2, int(0.012 * fs))), ir], axis=1)
    ir /= math.sqrt(float(np.sum(ir * ir)) / 2.0)
    _IR_CACHE[key] = ir
    return ir


# MEMORY. Every stage after synthesis used to materialise several full-length
# float64 temporaries of the whole stereo signal at once: rendering a 23-minute
# ADL piece (Liszt, Second Ballade; a 0.94 GB output) peaked at 8.4 GB traced,
# 4.7 GB of it in one giant reverb FFT and 5.6 GB in the loudness stage, whose
# limiter loop allocated ~6 signal-sized arrays per iteration. On a machine
# that is also training, that is the same silent swap cliff as the VRAM spill.
# The stages below therefore work in chunks of _CHUNK_BLOCKS 400 ms blocks and
# the reverb is overlap-add. Every chunk boundary is a multiple of the 400 ms
# gating block, so the gated loudness -- the only statistic the normaliser
# steers by -- is computed from exactly the same per-block sums as before, and
# the limiter/compressor apply exactly the same per-sample arithmetic.
_CHUNK_BLOCKS = 64            # 400 ms blocks per chunk (~25.6 s of audio)
_OLA_BLOCK = 1 << 18          # reverb overlap-add segment, samples


def _chunk_len(fs: int) -> int:
    return max(1, int(0.4 * fs)) * _CHUNK_BLOCKS


def _convolve(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Linear convolution of each row of ``x`` with ``ir`` (same leading shape).

    Overlap-add in segments of _OLA_BLOCK samples: one full-length FFT of a
    long piece needs ~5x the signal in complex temporaries (4.7 GB for a
    23-minute render) and is slower than many cache-sized ones.
    """
    m, L = x.shape[-1], ir.shape[-1]
    n = m + L - 1
    if m <= _OLA_BLOCK:
        nfft = _fast_len(n)
        return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[..., :n]
    nfft = _fast_len(_OLA_BLOCK + L - 1)
    H = np.fft.rfft(ir, nfft)
    out = np.zeros(x.shape[:-1] + (n,))
    for s in range(0, m, _OLA_BLOCK):
        seg = x[..., s: s + _OLA_BLOCK]
        k = seg.shape[-1] + L - 1
        out[..., s: s + k] += np.fft.irfft(np.fft.rfft(seg, nfft) * H, nfft)[..., :k]
    return out


def _chunk_stats(chunks, fs: int) -> Tuple[np.ndarray, float, int, float]:
    """(per-400ms-block mean powers, sum of per-sample power, n, sample peak).

    ``chunks`` yields (channels, m) float64 arrays whose lengths are multiples
    of the block except for the last, so each block's mean is computed over
    exactly the same contiguous samples as a whole-signal reshape would be.
    """
    blk = int(0.4 * fs)
    blocks, total, n, peak = [], 0.0, 0, 0.0
    for c in chunks:
        if c.size:
            peak = max(peak, float(np.abs(c).max()))
        power = np.mean(c * c, axis=0)
        total += float(power.sum())
        n += power.size
        full = (power.size // blk) * blk
        if full:
            blocks.append(power[:full].reshape(-1, blk).mean(axis=1))
    bp = np.concatenate(blocks) if blocks else np.zeros(0)
    return bp, total, n, peak


def _gated(bp: np.ndarray) -> Optional[float]:
    """BS.1770-style gating of block powers; None when there is no full block."""
    if bp.size == 0:
        return None
    bp = bp[bp > 1e-7]
    if bp.size:
        bp = bp[bp > bp.mean() * 0.01]
    return float(bp.mean()) if bp.size else 0.0


def _db(p: float) -> float:
    return 10.0 * math.log10(p) if p > 0 else float("-inf")


def _iter_chunks(x: np.ndarray, fs: int):
    step = _chunk_len(fs)
    for a in range(0, max(x.shape[-1], 1), step):
        yield x[:, a: a + step]


def loudness_stats(x: np.ndarray, fs: int) -> Dict[str, float]:
    """Gated RMS, plain RMS and sample peak of ``x`` in dBFS.

    ``x`` is mono or (channels, n). Gated RMS borrows the BS.1770 gating idea
    (without K-weighting, so it is RMS, not LUFS): 400 ms blocks, drop blocks
    below -70 dBFS, then blocks more than 20 dB below the mean of the rest.
    That ignores silences and the reverb tail. Computed in chunks, so it needs
    no signal-sized temporaries; int16 input is scaled by 1/32768 per chunk.
    """
    x = np.atleast_2d(np.asarray(x))
    if x.dtype == np.int16:
        chunks = (c.astype(np.float64) / 32768.0 for c in _iter_chunks(x, fs))
    else:
        chunks = (np.asarray(c, dtype=np.float64) for c in _iter_chunks(x, fs))
    bp, total, n, peak = _chunk_stats(chunks, fs)
    plain = total / n if n else 0.0
    gated = _gated(bp)
    return {
        "rms_db": _db(plain if gated is None else gated),
        "plain_rms_db": _db(plain),
        "peak_db": 20.0 * math.log10(peak) if peak > 0 else float("-inf"),
    }


def _block_reduce(values: np.ndarray, B: int, how: str) -> np.ndarray:
    """Per-block max or mean of ``values`` zero-padded to a multiple of ``B``.

    Same result as ``np.pad(values).reshape(-1, B).<how>(axis=1)`` -- every
    row is reduced over exactly the same B contiguous numbers -- without the
    signal-sized padded copy.
    """
    n = values.shape[-1]
    full = n // B
    head = values[: full * B].reshape(full, B)
    out = head.max(axis=1) if how == "max" else head.mean(axis=1)
    if n > full * B:
        last = np.zeros(B)
        last[: n - full * B] = values[full * B:]
        out = np.concatenate([out, [last.max() if how == "max" else last.mean()]])
    return out


def _interp_gain_db(n: int, centres: np.ndarray, G: np.ndarray, step: int) -> np.ndarray:
    """``10 ** (interp(arange(n), centres, G) / 20)`` built a chunk at a time."""
    gain = np.empty(n)
    for a in range(0, n, step):
        b = min(a + step, n)
        gain[a:b] = 10.0 ** (np.interp(np.arange(a, b), centres, G) / 20.0)
    return gain


def _limiter_gain(env: np.ndarray, fs: int, ceiling_db: float = _CEILING_DB,
                  lookahead: float = 0.005, release_db_per_s: float = 80.0) -> np.ndarray:
    """Per-sample limiter gain for a channel-linked peak envelope ``env``."""
    ceiling = 10.0 ** (ceiling_db / 20.0)
    n = env.shape[-1]
    B = max(1, int(lookahead * fs))
    nb = -(-n // B)
    blockpk = _block_reduce(env, B, "max")
    g = np.minimum(0.0, 20.0 * np.log10(ceiling / np.maximum(blockpk, 1e-12)))
    gn = g.copy()
    gn[1:] = np.minimum(gn[1:], g[:-1])
    gn[:-1] = np.minimum(gn[:-1], g[1:])
    r = release_db_per_s * B / fs
    k = np.arange(nb)
    G = k * r + np.minimum.accumulate(gn - k * r)
    return _interp_gain_db(n, (k + 0.5) * B, G, _chunk_len(fs))


def _limit(x: np.ndarray, fs: int, ceiling_db: float = _CEILING_DB,
           lookahead: float = 0.005, release_db_per_s: float = 80.0) -> np.ndarray:
    """Look-ahead, channel-linked peak limiter; output never exceeds the ceiling.

    Gain is computed per 5 ms block (min with both neighbours = look-ahead),
    recovers at a fixed dB/s (vectorised with minimum.accumulate), and is
    interpolated between block centres. Each interpolated value lies between
    two block gains that are both <= what the sample's own block needs, so
    no sample can overshoot.
    """
    ceiling = 10.0 ** (ceiling_db / 20.0)
    if x.shape[-1] == 0:
        return x
    gain = _limiter_gain(np.abs(x).max(axis=0), fs, ceiling_db, lookahead, release_db_per_s)
    return np.clip(x * gain, -ceiling, ceiling)


def _compress_gain(x: np.ndarray, fs: int, threshold_db: float, ratio: float = 2.0,
                   attack: float = 0.010, release: float = 0.200,
                   block: float = 0.005) -> np.ndarray:
    """Per-sample gain of the compressor described in :func:`_compress`."""
    n = x.shape[-1]
    B = max(1, int(block * fs))
    nb = -(-n // B)
    step = B * max(1, _chunk_len(fs) // B)           # chunks end on block edges
    means = []
    for a in range(0, n, step):                      # per-block mean power, chunkwise
        c = x[:, a: a + step]
        means.append(_block_reduce(np.mean(c * c, axis=0), B, "mean"))
    level = 10.0 * np.log10(np.maximum(np.concatenate(means) if means else np.zeros(0), 1e-12))
    over = level - threshold_db
    knee = 6.0
    target = np.where(over <= -knee / 2, 0.0,
                      np.where(over >= knee / 2, over,
                               (over + knee / 2) ** 2 / (2 * knee)))
    target = -target * (1.0 - 1.0 / ratio)                      # desired gain, dB (<= 0)
    a_att = math.exp(-block / attack)
    a_rel = math.exp(-block / release)
    g = np.empty(nb)
    cur = 0.0
    for i, t in enumerate(target.tolist()):
        coef = a_att if t < cur else a_rel
        cur = coef * cur + (1.0 - coef) * t
        g[i] = cur
    return _interp_gain_db(n, (np.arange(nb) + 0.5) * B, g, _chunk_len(fs))


def _compress(x: np.ndarray, fs: int, threshold_db: float, ratio: float = 2.0,
              attack: float = 0.010, release: float = 0.200, block: float = 0.005) -> np.ndarray:
    """Gentle soft-knee RMS compressor (channel-linked).

    Levels more than ``threshold_db`` get ``ratio``:1 compression, smoothed
    with attack/release time constants. It takes the edge off loud chords so
    the brickwall limiter after it only has to catch the last few dB -- a
    limiter doing 7+ dB alone audibly flattens piano attacks.
    The recursion runs per 5 ms block (not per sample), which is cheap.
    """
    return x * _compress_gain(x, fs, threshold_db, ratio, attack, release, block)


def _normalise(x: np.ndarray, fs: int, target_db: float, inplace: bool = False) -> np.ndarray:
    """Gain to ``target_db`` gated RMS, compress + limit, iterate to undo their loss.

    Numerically this is ``y = _limit(x' * s)`` iterated on the scalar ``s``,
    where ``x'`` is the compressed signal. It is evaluated without building
    ``x' * s`` or ``y`` in full on each iteration: the limiter only needs the
    peak envelope, and ``|x' * s| == |x'| * s`` exactly in floating point, so
    the envelope is computed once; the loudness of the limited signal is then
    measured chunk by chunk, and the final ``y`` overwrites ``x`` chunk by
    chunk. With ``inplace`` the caller's array is used as that buffer.
    """
    cur = loudness_stats(x, fs)["rms_db"]
    if not math.isfinite(cur):
        return x
    if inplace:
        x *= 10.0 ** ((target_db - cur) / 20.0)
    else:
        x = x * 10.0 ** ((target_db - cur) / 20.0)
    x *= _compress_gain(x, fs, target_db + 4.0)
    ceiling = 10.0 ** (_CEILING_DB / 20.0)
    n = x.shape[-1]
    step = _chunk_len(fs)
    env = np.empty(n)
    for a in range(0, n, step):
        env[a: a + step] = np.abs(x[:, a: a + step]).max(axis=0)

    def limited(scale: float, gain: np.ndarray):
        for a in range(0, max(n, 1), step):
            yield np.clip((x[:, a: a + step] * scale) * gain[a: a + step], -ceiling, ceiling)

    gain_db = target_db - loudness_stats(x, fs)["rms_db"]
    scale, gain = 1.0, None
    for _ in range(8):
        scale = 10.0 ** (gain_db / 20.0)
        gain = _limiter_gain(env * scale, fs) if n else np.zeros(0)
        bp, total, count, _peak = _chunk_stats(limited(scale, gain), fs)
        gated = _gated(bp)
        level = _db((total / count if count else 0.0) if gated is None else gated)
        err = target_db - level
        if abs(err) < 0.05:
            break
        gain_db += err
    # Each output chunk depends only on the same chunk of x (the gain is
    # already computed), so the result can overwrite x as it is produced.
    for a, chunk in zip(range(0, max(n, 1), step), limited(scale, gain)):
        x[:, a: a + chunk.shape[-1]] = chunk
    return x


def _collect_notes(midi: pretty_midi.PrettyMIDI):
    notes = [nt for inst in midi.instruments if not inst.is_drum for nt in inst.notes]
    starts = np.array([nt.start for nt in notes], dtype=np.float64)
    ends = np.array([nt.end for nt in notes], dtype=np.float64)
    pitches = np.array([nt.pitch for nt in notes], dtype=np.int64)
    vels = np.array([nt.velocity for nt in notes], dtype=np.int64)
    order = np.lexsort((pitches, starts))
    return starts[order], ends[order], pitches[order], vels[order]


def _legacy_organ(midi: pretty_midi.PrettyMIDI, fs: int) -> np.ndarray:
    """The original renderer's tone (pretty_midi.synthesize, 4-harmonic stack)."""
    def voice(phase):
        return (np.sin(phase) + 0.50 * np.sin(2 * phase)
                + 0.25 * np.sin(3 * phase) + 0.12 * np.sin(4 * phase))
    mono = midi.synthesize(fs=fs, wave=voice)
    return np.vstack([mono, mono])


def _synth_notes(starts, ends, pitches, vels, midi, fs, instrument, preset,
                 velocity_shaping, seed) -> np.ndarray:
    if velocity_shaping not in ("auto", "on", "off"):
        raise ValueError("velocity_shaping must be 'auto', 'on' or 'off'")
    if velocity_shaping == "on" or (velocity_shaping == "auto" and np.unique(vels).size <= 1):
        vels = shape_velocities(starts, pitches, vels, midi=midi, seed=seed)

    # how many notes are sounding at each onset (the note itself included)
    started = np.searchsorted(starts, starts + 0.02, side="right")
    finished = np.searchsorted(np.sort(ends), starts, side="right")
    poly_gain = np.maximum(1, started - finished).astype(np.float64) ** -preset.poly_exponent

    release_len = int(6.9 * preset.release * fs)        # damper tail to -60 dB
    # Pass 1: where each note goes and how long it is, so the mix can be
    # allocated once and every note added straight into it. The old code kept
    # every note's waveform in a list until the end (2.2 GB peak for a
    # 23-minute piece); the summation order -- note order -- is unchanged, so
    # the mix is bit-identical.
    specs: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    layout = []
    total = 0
    for s, e, p, v, pg in zip(starts, ends, pitches, vels, poly_gain):
        p, v = int(p), int(v)
        spec = specs.get((p, v))
        if spec is None:
            spec = specs[(p, v)] = partial_spec(p, v, instrument, fs)
        freqs, amps, rates = spec
        if amps.size == 0:
            continue
        natural = 6.9 / float(rates.min())               # string alone reaches -60 dB
        sustain = min(max(float(e - s), preset.min_sound), natural)
        n_off = max(1, int(round(sustain * fs)))
        i0 = int(round(s * fs))
        layout.append((i0, p, v, n_off, float(pg)))
        total = max(total, i0 + n_off + release_len)

    mix = np.zeros((2, total))
    # Waveforms are reused for repeated (pitch, velocity, length) notes. The
    # cache is bounded (LRU by bytes): unbounded it held 1.2 GB of unique
    # waveforms on the same piece. Evicted entries are recomputed identically.
    cache: "OrderedDict[Tuple[int, int, int], np.ndarray]" = OrderedDict()
    cached_bytes = 0
    for i0, p, v, n_off, pg in layout:
        key = (p, v, n_off)
        wav = cache.get(key)
        if wav is None:
            freqs, amps, rates = specs[(p, v)]
            n = n_off + release_len
            wav = _decaying_partials(freqs, amps, rates, n, fs)
            vn = v / 127.0
            if preset.hammer > 0:                        # felt-on-string thump
                nh = min(n, int(0.03 * fs))
                width = int(np.clip(fs / (4.0 * _hz(p) + 2000.0), 1, 8))
                c = np.cumsum(np.random.default_rng((p, v)).standard_normal(nh + width))
                thump = (c[width:] - c[:-width])[:nh] / width
                wav[:nh] += preset.hammer * (0.3 + vn) * thump * np.exp(-np.arange(nh) / (0.004 * fs))
            att = preset.attack[0] + (preset.attack[1] - preset.attack[0]) * (1.0 - vn)
            na = min(n, max(1, int(att * fs)))
            wav[:na] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(na) / na)
            if n > n_off:
                wav[n_off:] *= np.exp(-np.arange(n - n_off) / (preset.release * fs))
            wav *= vn ** 1.6                             # ~ -9.5 dB at velocity 64
            cache[key] = wav
            cached_bytes += wav.nbytes
            while cached_bytes > _WAV_CACHE_BYTES and len(cache) > 1:
                cached_bytes -= cache.popitem(last=False)[1].nbytes
        else:
            cache.move_to_end(key)
        theta = (preset.pan_width * float(np.clip((p - 60) / 30.0, -1.0, 1.0)) + 1.0) * np.pi / 4.0
        gl = math.sqrt(2) * math.cos(theta) * pg
        gr = math.sqrt(2) * math.sin(theta) * pg
        mix[0, i0: i0 + wav.size] += gl * wav
        mix[1, i0: i0 + wav.size] += gr * wav
    return mix


def synthesize_midi(
    midi: Union[pretty_midi.PrettyMIDI, str, Path],
    fs: int = DEFAULT_SAMPLE_RATE,
    instrument: str = "piano",
    reverb: Optional[float] = None,
    velocity_shaping: str = "auto",
    seed: int = 0,
    target_rms_db: Optional[float] = DEFAULT_TARGET_RMS_DB,
    soundfont: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Render MIDI to a float stereo array of shape (2, n), within [-1, 1].

    Args:
        midi: A PrettyMIDI object or a path to a .mid file.
        fs: Sample rate in Hz.
        instrument: ``piano`` (default), ``epiano``, or ``organ`` (the old sound).
        reverb: Wet mix 0..1. ``None`` uses the preset default; 0 disables.
        velocity_shaping: ``auto`` adds dynamics only when every note has the
            same velocity (i.e. raw model output); ``on`` always; ``off`` never.
        seed: Seed for the velocity jitter. Output is fully deterministic.
        target_rms_db: Gated-RMS target in dBFS; peaks are limited to -1 dBFS.
            ``None`` returns the raw, unnormalised mix.
        soundfont: Optional .sf2 path. Renders with FluidSynth instead of the
            built-in synth; needs the FluidSynth library and ``pyfluidsynth``.
    """
    if not isinstance(midi, pretty_midi.PrettyMIDI):
        midi = pretty_midi.PrettyMIDI(str(midi))
    if instrument not in INSTRUMENTS:
        raise ValueError(f"unknown instrument {instrument!r}; choose from {sorted(INSTRUMENTS)}")
    preset = INSTRUMENTS[instrument]
    wet = preset.reverb if reverb is None else float(reverb)
    starts, ends, pitches, vels = _collect_notes(midi)

    if soundfont is not None:
        try:
            import fluidsynth  # noqa: F401  - pretty_midi.fluidsynth needs it
        except ImportError as exc:
            raise RuntimeError(
                "--soundfont needs FluidSynth: install the FluidSynth library "
                "(https://www.fluidsynth.org) and `pip install pyfluidsynth`, "
                "or leave out --soundfont to use the built-in synth."
            ) from exc
        sf2 = Path(soundfont)
        if not sf2.is_file():
            raise FileNotFoundError(f"soundfont not found: {sf2}")
        mono = midi.fluidsynth(fs=fs, sf2_path=str(sf2)) if starts.size else np.zeros(0)
        mix = np.vstack([mono, mono]).astype(np.float64)
        wet = 0.0  # FluidSynth applies its own reverb
    elif starts.size == 0:
        mix = np.zeros((2, 0))
    elif instrument == "organ":
        mix = _legacy_organ(midi, fs)
    else:
        mix = _synth_notes(starts, ends, pitches, vels, midi, fs, instrument, preset,
                           velocity_shaping, seed)

    if mix.shape[1] == 0:
        return np.zeros((2, int(0.25 * fs)))    # empty MIDI: a short silence

    if wet > 0:
        # (1 - wet) * dry + wet * room, formed in place in the (longer) room
        # buffer instead of padding a copy of the dry mix and adding two new
        # full-length products; addition is commutative, so the samples match.
        room = _convolve(mix, _room_ir(fs))
        room *= wet
        step = _chunk_len(fs)
        dry = mix.shape[1]
        for a in range(0, dry, step):
            b = min(a + step, dry)
            room[:, a:b] += (1.0 - wet) * mix[:, a:b]
        mix = room
        del room
        env = np.empty(mix.shape[1])             # drop the tail below -80 dB of peak
        for a in range(0, mix.shape[1], step):
            env[a: a + step] = np.abs(mix[:, a: a + step]).max(axis=0)
        audible = np.flatnonzero(env > env.max() * 1e-4)
        if audible.size:
            mix = mix[:, : int(audible[-1]) + 1]

    if target_rms_db is not None:
        mix = _normalise(mix, fs, target_rms_db, inplace=True)

    fade = min(int(0.02 * fs), mix.shape[1] // 2)
    if fade > 0:
        mix[:, -fade:] *= np.linspace(1.0, 0.0, fade)
    return mix


@dataclass
class RenderStats:
    """What ``render_wav`` wrote, measured on the final 16-bit samples."""

    path: Path
    seconds: float
    rms_db: float          # gated RMS, dBFS
    plain_rms_db: float    # whole-file RMS, dBFS
    peak_db: float         # sample peak, dBFS
    render_seconds: float
    notes: int


def render_wav(
    midi: Union[pretty_midi.PrettyMIDI, str, Path],
    out_path: Union[str, Path],
    fs: int = DEFAULT_SAMPLE_RATE,
    **kwargs,
) -> RenderStats:
    """Render MIDI to a 16-bit stereo WAV. ``kwargs`` go to ``synthesize_midi``."""
    t0 = time.perf_counter()
    if not isinstance(midi, pretty_midi.PrettyMIDI):
        midi = pretty_midi.PrettyMIDI(str(midi))
    audio = synthesize_midi(midi, fs=fs, **kwargs)
    # Straight into interleaved int16, a chunk at a time: the old
    # clip(rint(audio * 32767)) chain plus the transposed copy cost four more
    # signal-sized temporaries.
    n = audio.shape[1]
    frames = np.empty((n, 2), dtype="<i2")
    step = _chunk_len(fs)
    for a in range(0, n, step):
        frames[a: a + step] = np.clip(np.rint(audio[:, a: a + step] * 32767.0), -32768, 32767).T
    del audio
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(fs)
        fh.writeframes(frames.tobytes())
    elapsed = time.perf_counter() - t0
    stats = loudness_stats(frames.T, fs)
    return RenderStats(
        path=out, seconds=n / fs, rms_db=stats["rms_db"],
        plain_rms_db=stats["plain_rms_db"], peak_db=stats["peak_db"],
        render_seconds=elapsed,
        notes=sum(len(i.notes) for i in midi.instruments if not i.is_drum),
    )
