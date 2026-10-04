"""Tests for the training-efficiency options: adaptive softmax, TF32, fp16,
torch.compile fallback, LR schedules, min_delta / patience, regularisation.

CPU only and tiny models, like tests/test_model_audit.py. The GPU speed and
loss-parity numbers these options were adopted (or rejected) on live in
configs/default.yaml; these tests pin correctness, not speed.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.types import RESERVED_SYMBOLS  # noqa: E402
from src.data.vocab import Vocab  # noqa: E402
from src.models import MusicLSTM, build_model  # noqa: E402
from src.models.lstm import sanitize_cutoffs  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import generate as G  # noqa: E402
from src import train as T  # noqa: E402

V = 60


def adaptive_lstm(vocab_size: int = V, seed: int = 0, cutoffs=(8, 30), counts=None) -> MusicLSTM:
    torch.manual_seed(seed)
    model = MusicLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=2, dropout=0.0,
                      output="adaptive", adaptive_cutoffs=cutoffs)
    if counts is None:
        # Deliberately NOT aligned with id order, so the permutation matters.
        counts = np.random.RandomState(seed).randint(0, 1000, vocab_size)
    model.set_token_frequencies(counts)
    return model.eval()


# --------------------------------------------------------------------------
# adaptive softmax: exactness
# --------------------------------------------------------------------------


def test_adaptive_log_probs_sum_to_one_over_the_vocabulary() -> None:
    model = adaptive_lstm()
    x = torch.randint(1, V, (4, 9))
    with torch.no_grad():
        logp = model(x)
    assert logp.shape == (4, 9, V)
    total = torch.logsumexp(logp.double(), dim=-1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-5)


def test_adaptive_loss_is_the_nll_of_its_own_distribution() -> None:
    """model.loss (cluster-sparse) must equal -log p(target) read off the full
    log_prob -- through the id<->rank permutation -- including ignore_index."""
    model = adaptive_lstm()
    x = torch.randint(1, V, (3, 7))
    y = torch.randint(1, V, (3, 7))
    y[0, :4] = 0
    with torch.no_grad():
        logp = model(x)
        ref = F.nll_loss(logp.reshape(-1, V), y.reshape(-1), ignore_index=0)
        got = model.loss(x, y, ignore_index=0)
    assert torch.allclose(got, ref, atol=1e-5)


def test_adaptive_loss_all_ignored_is_nan_like_cross_entropy() -> None:
    model = adaptive_lstm()
    x = torch.randint(1, V, (2, 5))
    with torch.no_grad():
        assert math.isnan(float(model.loss(x, torch.zeros_like(x), ignore_index=0)))


def test_frequency_permutation_puts_common_ids_in_the_head() -> None:
    counts = np.zeros(V, dtype=np.int64)
    counts[[50, 3, 17]] = [900, 800, 700]          # the three most common ids
    model = adaptive_lstm(counts=counts)
    assert model.rank_to_id[:3].tolist() == [50, 3, 17]
    assert model.id_to_rank[50] == 0 and model.id_to_rank[3] == 1 and model.id_to_rank[17] == 2
    # A true permutation, and the two buffers are inverses.
    assert sorted(model.rank_to_id.tolist()) == list(range(V))
    assert torch.equal(model.rank_to_id[model.id_to_rank], torch.arange(V))
    # Ties break by id: every zero-count id after the three, in id order.
    rest = [i for i in range(V) if i not in (50, 3, 17)]
    assert model.rank_to_id[3:].tolist() == rest


def test_permutation_is_saved_and_restored_with_the_weights(tmp_path) -> None:
    model = adaptive_lstm(seed=3)
    state = model.state_dict()
    assert "id_to_rank" in state and "rank_to_id" in state
    torch.save(state, tmp_path / "m.pt")
    fresh = MusicLSTM(V, embed_dim=16, hidden_dim=32, num_layers=2, dropout=0.0,
                      output="adaptive", adaptive_cutoffs=(8, 30)).eval()   # identity permutation
    fresh.load_state_dict(torch.load(tmp_path / "m.pt"))
    x = torch.randint(1, V, (2, 6))
    with torch.no_grad():
        assert torch.allclose(fresh(x), model(x))


def test_full_model_state_dict_is_unchanged() -> None:
    """Old checkpoints must keep loading: output=full adds no keys."""
    model = MusicLSTM(V, embed_dim=16, hidden_dim=32, num_layers=2)
    keys = set(model.state_dict())
    assert {"fc2.weight", "fc2.bias", "fc1.weight", "embedding.weight"} <= keys
    assert not any("rank" in k or "adaptive" in k for k in keys)


def test_full_model_loss_equals_the_old_logits_path() -> None:
    torch.manual_seed(0)
    model = MusicLSTM(V, embed_dim=16, hidden_dim=32, num_layers=2).eval()
    x = torch.randint(1, V, (3, 8))
    y = torch.randint(1, V, (3, 8))
    y[1, :2] = 0
    with torch.no_grad():
        old = nn.CrossEntropyLoss(ignore_index=0)(model(x).reshape(-1, V), y.reshape(-1))
        new = model.loss(x, y, ignore_index=0)
    assert torch.equal(old, new)


@pytest.mark.parametrize("output", ["full", "adaptive"])
def test_last_only_matches_the_last_row(output: str) -> None:
    model = adaptive_lstm() if output == "adaptive" else MusicLSTM(V, 16, 32, 2).eval()
    x = torch.randint(1, V, (2, 11))
    with torch.no_grad():
        assert torch.allclose(model(x, last_only=True)[:, 0], model(x)[:, -1], atol=1e-6)


def test_adaptive_incremental_decoding_matches_full_window() -> None:
    model = adaptive_lstm(seed=4)
    seed, seq_len = [5, 9, 11, 13], 40
    kwargs = dict(num_tokens=seq_len - len(seed) + 1, seq_len=seq_len, device=torch.device("cpu"),
                  temperature=1.0, top_k=0, top_p=0.0)
    full = G._sample_ids(model, seed, generator=torch.Generator().manual_seed(1), incremental=False, **kwargs)
    inc = G._sample_ids(model, seed, generator=torch.Generator().manual_seed(1), incremental=True, **kwargs)
    assert full == inc


def test_perplexity_uses_the_model_loss_path() -> None:
    model = adaptive_lstm()
    x = torch.randint(1, V, (6, 8))
    y = torch.randint(1, V, (6, 8))
    ppl = E.perplexity(model, DataLoader(TensorDataset(x, y), batch_size=4))   # uneven last batch
    with torch.no_grad():
        ref = F.nll_loss(model(x).reshape(-1, V), y.reshape(-1))
    assert math.isclose(ppl, math.exp(float(ref)), rel_tol=1e-5)


def test_cutoffs_are_clamped_for_small_vocabularies() -> None:
    assert sanitize_cutoffs((2000, 10000), 37020) == [2000, 10000]
    assert sanitize_cutoffs((2000, 10000), 5000) == [2000]
    assert sanitize_cutoffs((2000, 10000), 40) == [20]
    with pytest.raises(ValueError):
        sanitize_cutoffs((1,), 2)


def test_build_model_reads_the_output_knobs() -> None:
    cfg = load_config(**{"model.output": "adaptive", "model.adaptive_cutoffs": [10, 20],
                         "model.hidden_dim": 32, "model.embed_dim": 16})
    model = build_model(cfg, V)
    assert model.output == "adaptive" and model.adaptive.cutoffs[:-1] == [10, 20]
    with pytest.raises(ValueError, match="lstm only"):
        build_model(load_config(**{"model.arch": "transformer", "model.output": "adaptive"}), V)
    with pytest.raises(ValueError, match="model.output"):
        build_model(load_config(**{"model.output": "hierarchical"}), V)


def test_generate_builds_the_checkpoints_output_layer() -> None:
    """A run trained with --set model.output=adaptive is sampled with the plain
    config; the checkpoint's head fields must win, and old checkpoints (no
    `output` key) must stay full-softmax."""
    cfg = load_config()
    ckpt = {"config": {"model": {"output": "adaptive", "adaptive_cutoffs": [10, 20]}}}
    new = G._with_checkpoint_head(cfg, ckpt)
    assert new.model.output == "adaptive" and list(new.model.adaptive_cutoffs) == [10, 20]
    assert cfg.model.output == "full"                                   # caller's cfg untouched
    assert G._with_checkpoint_head(cfg, {"config": {"model": {}}}) is cfg
    adaptive_cfg = load_config(**{"model.output": "adaptive"})
    assert G._with_checkpoint_head(adaptive_cfg, {"config": {"model": {}}}).model.output == "full"


# --------------------------------------------------------------------------
# trainer knobs
# --------------------------------------------------------------------------


def test_default_patience_is_three() -> None:
    assert int(load_config().train.patience) == 3


def test_tf32_modes_set_and_leave_the_flags() -> None:
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        before = saved
        assert T.apply_tf32("default") == "default"
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
        assert T.apply_tf32("on") == "on"
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert T.apply_tf32(False) == "off"
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        with pytest.raises(ValueError):
            T.apply_tf32("sometimes")
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def test_amp_accepts_fp16_and_is_off_on_cpu() -> None:
    assert T._resolve_amp("fp16", torch.device("cpu")) is None
    assert T._resolve_amp("off", torch.device("cuda")) is None
    assert T._resolve_amp("fp16", torch.device("cuda")) is torch.float16
    with pytest.raises(ValueError):
        T._resolve_amp("fp8", torch.device("cuda"))


def test_compile_failure_falls_back_to_eager(monkeypatch, capsys) -> None:
    def broken(fn, *a, **k):
        def call(*args):
            raise RuntimeError("Cannot find a working triton installation")
        return call

    monkeypatch.setattr(T.torch, "compile", broken)
    fn = lambda a, b: a + b  # noqa: E731
    got, compiled = T.maybe_compile(fn, True, (torch.ones(1), torch.ones(1)))
    assert got is fn and not compiled
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "eager" in out
    assert T.maybe_compile(fn, False) == (fn, False)


def test_plateau_schedule_halves_lr_after_stale_epochs() -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = load_config(**{"train.lr_schedule": "plateau", "train.lr_patience": 1}).train
    sched, kind = T.build_scheduler(opt, cfg, steps_per_epoch=10)
    assert kind == "plateau"
    for val in (3.0, 2.9, 2.95, 2.96):      # improve, then two stale epochs
        sched.step(val)
    assert math.isclose(opt.param_groups[0]["lr"], 5e-4)


def test_cosine_schedule_decays_to_the_floor_over_all_steps() -> None:
    model = nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfg = load_config(**{"train.lr_schedule": "cosine", "train.epochs": 2, "train.min_lr_ratio": 0.1}).train
    sched, kind = T.build_scheduler(opt, cfg, steps_per_epoch=5)
    for _ in range(10):
        opt.step()
        sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1e-4, rel_tol=1e-6)
    with pytest.raises(ValueError):
        T.build_scheduler(opt, load_config(**{"train.lr_schedule": "step"}).train, 5)


def test_weight_decay_and_dropout_are_honoured() -> None:
    cfg = load_config(**{"train.weight_decay": 0.01, "train.optimizer": "adamw",
                         "model.dropout": 0.45, "model.hidden_dim": 32, "model.embed_dim": 16})
    model = build_model(cfg, V)
    assert model.dropout.p == 0.45 and model.lstm.dropout == 0.45
    opt = T.build_optimizer(model, cfg.train)
    assert isinstance(opt, torch.optim.AdamW)
    assert all(g["weight_decay"] == 0.01 for g in opt.param_groups)


def _tiny_vocab() -> Vocab:
    return Vocab(list(RESERVED_SYMBOLS) + [str(p) for p in range(40, 80)])


def _train_cfg(tmp_path: Path, **extra):
    return load_config(**{
        "name": "eff_train",
        "model.embed_dim": 16, "model.hidden_dim": 32, "model.num_layers": 2,
        "model.seq_len": 8, "train.device": "cpu", "train.epochs": 3,
        "train.checkpoint_dir": str(tmp_path / "runs"),
        **extra,
    })


def _fake_loaders(monkeypatch, vocab: Vocab) -> None:
    g = torch.Generator().manual_seed(0)
    n = len(vocab)

    def make(rows):
        data = torch.randint(5, n, (rows, 9), generator=g)
        return DataLoader(TensorDataset(data[:, :-1], data[:, 1:]), batch_size=16)

    loaders = (make(40), make(16), make(4), vocab)
    monkeypatch.setattr(T, "dataloaders_from_cache", lambda cfg: loaders)


def _history(run_dir: Path):
    with open(run_dir / "history.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_adaptive_model_trains_and_checkpoints_its_permutation(tmp_path, monkeypatch) -> None:
    vocab = _tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    T.train(_train_cfg(tmp_path, **{"model.output": "adaptive", "model.adaptive_cutoffs": [10, 25]}))
    ckpt = torch.load(tmp_path / "runs" / "eff_train" / "best.pt", weights_only=False)
    state = ckpt["model_state"]
    assert "id_to_rank" in state
    # Ids below 5 never occur in the fake data, so they rank last.
    assert set(state["rank_to_id"][-5:].tolist()) == set(range(5))
    assert ckpt["config"]["model"]["output"] == "adaptive"
    rows = _history(tmp_path / "runs" / "eff_train")
    assert all(math.isfinite(float(r["val_loss"])) for r in rows)


def test_plateau_schedule_is_logged_and_resumable(tmp_path, monkeypatch) -> None:
    vocab = _tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    # lr 0 never improves after epoch 1, so the plateau schedule must cut it.
    base = {"train.lr": 0.0, "train.lr_schedule": "plateau", "train.lr_patience": 0,
            "train.patience": 0}
    T.train(_train_cfg(tmp_path, **base))
    last = torch.load(tmp_path / "runs" / "eff_train" / "last.pt", weights_only=False)
    assert "scheduler_state" in last
    T.train(_train_cfg(tmp_path, **base, **{"train.epochs": 4, "train.resume": True}))
    assert [int(r["epoch"]) for r in _history(tmp_path / "runs" / "eff_train")] == [1, 2, 3, 4]


def test_cosine_lr_is_recorded_per_epoch(tmp_path, monkeypatch) -> None:
    vocab = _tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    T.train(_train_cfg(tmp_path, **{"train.lr_schedule": "cosine", "train.patience": 0}))
    lrs = [float(r["lr"]) for r in _history(tmp_path / "runs" / "eff_train")]
    assert lrs[0] == pytest.approx(1e-3) and lrs[0] > lrs[1] > lrs[2]


def test_min_delta_ignores_tiny_improvements(tmp_path, monkeypatch) -> None:
    vocab = _tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    # A tiny LR moves val loss by far less than 1.0 per epoch: with min_delta 1
    # only epoch 1 (from +inf) counts, and patience 2 stops after epoch 3.
    T.train(_train_cfg(tmp_path, **{"train.lr": 1e-4, "train.epochs": 10, "train.patience": 2,
                                    "train.min_delta": 1.0}))
    rows = _history(tmp_path / "runs" / "eff_train")
    assert [int(r["improved"]) for r in rows] == [1, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast needs CUDA")
def test_adaptive_softmax_survives_fp16_autocast() -> None:
    """BUG: nn.AdaptiveLogSoftmaxWithLoss under fp16 autocast raised
    'index_copy_(): self and source expected to have the same dtype'."""
    model = adaptive_lstm().cuda().train()
    x = torch.randint(1, V, (2, 6), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model.loss(x, x, ignore_index=0)
        logp = model(x, last_only=True)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert torch.allclose(logp.float().exp().sum(-1), torch.ones(2, 1, device="cuda"), atol=1e-4)
