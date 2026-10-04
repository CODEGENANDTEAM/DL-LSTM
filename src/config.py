"""Config loading. One YAML file fully describes an experiment.

The config hash is written into every checkpoint and every processed-data
cache directory, so a model can never be silently paired with a vocabulary or
dataset built under different settings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access, so cfg.model.hidden_dim works.

    A nested section is wrapped once and stored back, so ``cfg.model`` is the
    same object on every access. It used to be a fresh copy each time, which
    made ``cfg.model.seq_len = 7`` write into a temporary and vanish silently.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML config, layering it over configs/default.yaml.

    Overrides use dotted keys: ``load_config(p, **{"model.hidden_dim": 256})``.
    """
    base = _read_yaml(DEFAULT_CONFIG)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        base = _deep_merge(base, _read_yaml(path))
    for dotted, value in overrides.items():
        _set_dotted(base, dotted, value)
    return Config(base)


def config_hash(cfg: Dict[str, Any], *sections: str) -> str:
    """Stable short hash of the given sections (all sections if none given).

    Used to key the processed-data cache: change the encoding scheme or the
    quantization grid and you get a fresh cache directory instead of a
    silently stale one.
    """
    subset = {k: cfg[k] for k in sections if k in cfg} if sections else dict(cfg)
    blob = json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_override_value(text: str) -> Any:
    """Parse one ``--set KEY=VALUE`` value.

    YAML typing, so ``256`` is an int, ``false`` a bool, ``[-3,3]`` a list and
    ``null`` None -- plus one fix-up: PyYAML implements YAML 1.1, where a float
    needs a dot, so ``1e-4`` came back as the STRING "1e-4". Every CLI docstring
    advertised ``--set train.lr=1e-4``; anything that did arithmetic on such a
    value (``data.grid``, say) would crash or, worse, hash differently from the
    float. Exactly the exponent forms YAML 1.1 misses (``1e-4``, ``2.5e1``) are
    therefore returned as floats -- and only those: the earlier ``float(value)``
    fallback also turned ``nan``, ``inf`` and ``Infinity`` into floats, so e.g.
    ``--set name=nan`` produced a NaN name, and a NaN never equals itself.
    """
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str) and _EXPONENT_FLOAT.fullmatch(value.strip()):
        return float(value)
    return value


# A decimal number with an exponent: what YAML 1.1 fails to type as a float.
_EXPONENT_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    # Both sides are copied: assigning `over`'s lists/dicts by reference let a
    # later mutation of the merged config reach back into the caller's dict.
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_dotted(target: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    if not all(keys):
        raise ValueError(f"malformed override key {dotted!r}")
    for depth, key in enumerate(keys[:-1]):
        child = target.setdefault(key, {})
        if child is None:
            child = target[key] = {}
        if not isinstance(child, dict):
            prefix = ".".join(keys[: depth + 1])
            raise ValueError(
                f"cannot set {dotted!r}: {prefix!r} is {child!r}, not a section"
            )
        target = child
    target[keys[-1]] = value


# --------------------------------------------------------------------------
# GPU memory estimate
#
# On Windows a CUDA allocation that does not fit in VRAM does not fail: the
# driver spills to system RAM over PCIe and every step silently gets 20-40x
# slower (see configs/adl_mixed.yaml: batch 128 at vocab 25k took 1,392 s per
# epoch instead of ~70 s). The dominant term for this model family is the
# output layer: logits [B, L, V] exist as the fp32 forward output, the
# log-softmax CrossEntropyLoss saves for backward, and their gradient -- about
# three copies. This estimate lets a config be checked before the first epoch.
# --------------------------------------------------------------------------

# Calibrated on the RTX 5070 runs recorded in configs/adl_mixed.yaml
# (512h x 3L, embed 256, seq 256, vocab 37,847, fp32): measured 8.5 GB
# allocated at batch 64 and 6.5 GB at batch 48; this model gives 8.6 / 6.6.
_LOGIT_COPIES = 3.0
_LSTM_ACTIVATION_COPIES = 6.0    # 4 gates + h + c per layer and step, saved for backward
_PARAM_COPIES = 4.0              # weights, grads, Adam exp_avg, exp_avg_sq
# Allocated memory above this fraction of the card left too little for the
# caching allocator's reserve plus other processes (the desktop, a browser):
# batch 64 at 8.5 of 12.2 GB (70%) spilled in the real run, 6.5 GB (53%) did not.
DEFAULT_VRAM_BUDGET = 0.65


def estimate_train_memory(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden_dim: int = 512,
    num_layers: int = 3,
    embed_dim: int = 256,
    bytes_per_value: int = 4,
) -> Dict[str, float]:
    """Rough peak *allocated* GPU bytes for one LSTM training step.

    Returns ``{"logits", "activations", "params", "total"}`` in bytes. For a
    transformer the activation term is an underestimate, but the logits term
    -- the one that grows with the vocabulary -- is the same. Pure arithmetic:
    no torch, no device, so it can run anywhere, including in tests.
    """
    B, L, V = int(batch_size), int(seq_len), int(vocab_size)
    H, E, N = int(hidden_dim), int(embed_dim), max(1, int(num_layers))
    logits = _LOGIT_COPIES * B * L * V * bytes_per_value
    activations = _LSTM_ACTIVATION_COPIES * N * B * L * H * bytes_per_value
    params = (
        V * E                                        # embedding
        + 4 * H * (E + H) + 8 * H                    # first LSTM layer
        + (N - 1) * (4 * H * (2 * H) + 8 * H)        # further layers
        + H * V + V                                  # output projection
    )
    param_bytes = _PARAM_COPIES * params * 4         # params/optimizer stay fp32
    return {
        "logits": float(logits),
        "activations": float(activations),
        "params": float(param_bytes),
        "total": float(logits + activations + param_bytes),
    }


def gpu_memory_warning(
    cfg: Dict[str, Any],
    vocab_size: int,
    device_total_bytes: int,
    budget: float = DEFAULT_VRAM_BUDGET,
) -> "str | None":
    """A warning message if ``cfg`` likely exceeds GPU memory, else None.

    How the trainer should call it (src/train.py, after the vocab is loaded
    and the device resolved, before the first epoch)::

        if device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            message = gpu_memory_warning(cfg, len(vocab), total)
            if message:
                LOGGER.warning(message)   # or print(); do not raise

    ``bf16`` autocast halves the logits term; pass the config as-is and the
    estimate stays conservative.
    """
    model = cfg.get("model", {}) or {}
    train = cfg.get("train", {}) or {}
    batch = int(train.get("batch_size", 64))
    seq_len = int(model.get("seq_len", 100))
    est = estimate_train_memory(
        batch, seq_len, vocab_size,
        hidden_dim=int(model.get("hidden_dim", 512)),
        num_layers=int(model.get("num_layers", 3)),
        embed_dim=int(model.get("embed_dim", 256)),
    )
    if device_total_bytes <= 0 or est["total"] <= budget * device_total_bytes:
        return None
    per_sample = (est["logits"] + est["activations"]) / max(1, batch)
    fixed = est["params"]
    safe = int((budget * device_total_bytes - fixed) // per_sample) if per_sample else batch
    gb = 1e9
    return (
        f"estimated GPU memory {est['total'] / gb:.1f} GB (logits {est['logits'] / gb:.1f} GB = "
        f"{batch} x {seq_len} x {vocab_size} x 4 B x {_LOGIT_COPIES:g}) exceeds {budget:.0%} of "
        f"the card's {device_total_bytes / gb:.1f} GB. On Windows the driver then spills to "
        f"system RAM and training runs 20-40x slower without an error. Reduce "
        f"train.batch_size to <= {max(1, safe)}, shorten model.seq_len, or cap the "
        f"vocabulary (encoding.max_size / min_freq)."
    )
