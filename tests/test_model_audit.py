"""Regression tests for the model / training / generation / CLI audit.

Every test here either pins a bug that was found and fixed (the docstring says
what the bug was) or guards an edge case that the fix relies on. All of them run
on CPU with tiny models, so the suite stays fast and needs no checkpoint.
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.types import RESERVED_SYMBOLS, NoteEvent, Piece  # noqa: E402
from src.data.vocab import Vocab  # noqa: E402
from src.models import MusicLSTM, MusicTransformer  # noqa: E402
from src import generate as G  # noqa: E402
from src import sampling as S  # noqa: E402
from src import train as T  # noqa: E402
from src import evaluate as E  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

NOTE_SYMBOLS = ["60", "62", "64", "65", "67", "69", "71", "72", "<REST:2>"]
STYLE_SYMBOLS = ["<STYLE:Blues>", "<STYLE:Jazz>"]


def tiny_vocab() -> Vocab:
    return Vocab(list(RESERVED_SYMBOLS) + NOTE_SYMBOLS + STYLE_SYMBOLS)


def tiny_lstm(vocab_size: int, seed: int = 0) -> MusicLSTM:
    torch.manual_seed(seed)
    model = MusicLSTM(vocab_size, embed_dim=16, hidden_dim=32, num_layers=2, dropout=0.3)
    return model.eval()


# --------------------------------------------------------------------------
# sampling.py
# --------------------------------------------------------------------------


def test_top_p_keeps_the_smallest_nucleus_at_an_exact_boundary() -> None:
    """BUG: with probs [.5, .25, .25] and p=.75 the nucleus is {0, 1}; the old
    `cumulative > p` test plus shift also kept token 2 whenever the running
    total hit p exactly."""
    logits = torch.tensor([0.5, 0.25, 0.25]).log()
    out = S.top_p_filter(logits, 0.75)
    assert torch.isfinite(out[:2]).all()
    assert out[2] == float("-inf")


def test_top_p_always_keeps_top_token_when_it_alone_exceeds_p() -> None:
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    out = S.top_p_filter(logits, 0.5)
    assert torch.isfinite(out[0]) and torch.isinf(out[1:]).all()


def test_nan_logits_never_sample_the_nan_token() -> None:
    """BUG: a NaN logit made softmax all-NaN and the fallback was argmax of the
    raw row -- which torch resolves to the NaN index itself."""
    logits = torch.tensor([0.0, 5.0, float("nan"), 1.0])
    for t in (0.0, 0.9):
        assert S.sample_next(logits, temperature=t) == 1


def test_banned_ids_are_never_sampled() -> None:
    logits = torch.tensor([9.0, 9.0, 0.0, 0.0])
    torch.manual_seed(0)
    draws = {S.sample_next(logits, temperature=1.0, banned_ids=[0, 1]) for _ in range(200)}
    assert draws <= {2, 3}
    assert S.sample_next(logits, temperature=0.0, banned_ids=[0]) == 1


@pytest.mark.parametrize("top_k,top_p,temp", [(10_000, 0.0, 1.0), (0, 1.0, 1.0), (0, 0.0, 1e-8), (1, 0.9, 0.5)])
def test_sampling_edge_settings_do_not_crash(top_k: int, top_p: float, temp: float) -> None:
    logits = torch.randn(50)
    torch.manual_seed(1)
    idx = S.sample_next(logits, temperature=temp, top_k=top_k, top_p=top_p)
    assert 0 <= idx < 50
    if temp < 1e-3 or top_k == 1:
        assert idx == int(torch.argmax(logits))


def test_inf_logit_is_chosen() -> None:
    logits = torch.tensor([0.0, float("inf"), 1.0])
    assert S.sample_next(logits, temperature=0.9) == 1


def test_sample_batch_rows_use_their_own_generators() -> None:
    """Row i of a batched draw must equal a solo draw with the same generator."""
    torch.manual_seed(0)
    logits = torch.randn(3, 40)
    gens = [torch.Generator().manual_seed(100 + i) for i in range(3)]
    batched = S.sample_batch(logits, temperature=0.9, generators=gens).tolist()
    solo = [
        S.sample_next(logits[i], temperature=0.9, generator=torch.Generator().manual_seed(100 + i))
        for i in range(3)
    ]
    assert batched == solo


# --------------------------------------------------------------------------
# generate.py -- seed windows, special tokens, incremental decoding
# --------------------------------------------------------------------------


def test_short_seed_is_not_left_padded_with_pad() -> None:
    """BUG: a seed shorter than seq_len was left-padded with <PAD>. Training
    windows never contain PAD (short pieces are dropped, not padded), so the
    model was primed on up to seq_len-1 tokens it had never seen as input."""
    assert G._window_in([5, 6, 7], 16, random.Random(0), pad=0) == [5, 6, 7]
    assert G._seed_context([5, 6, 7], 16) == [5, 6, 7]
    assert G._seed_context(list(range(1, 40)), 16) == list(range(24, 40))

    seen: List[torch.Tensor] = []

    class Spy(nn.Module):
        def forward(self, x, hidden=None, return_hidden=False):
            seen.append(x.clone())
            return torch.zeros(x.size(0), x.size(1), 12)

    G._sample_ids(Spy(), [5, 6, 7], num_tokens=3, seq_len=16, device=torch.device("cpu"),
                  temperature=0.9, top_k=0, top_p=0.0)
    assert seen and all((x != 0).all() for x in seen)
    assert seen[0].shape[1] == 3


def test_window_in_can_reach_the_final_window() -> None:
    """BUG: randrange(0, len-seq_len) excluded the last valid start offset."""
    seq = list(range(1, 12))          # 11 tokens, seq_len 10 -> starts 0 and 1
    starts = {G._window_in(seq, 10, random.Random(s), pad=0)[0] for s in range(64)}
    assert starts == {1, 2}


def test_special_token_ids_cover_pad_bos_eos_style_and_unk() -> None:
    vocab = tiny_vocab()
    banned = set(G.special_token_ids(vocab))
    names = {vocab.itos[i] for i in banned}
    assert {"<PAD>", "<BOS>", "<EOS>", "<UNK>", *STYLE_SYMBOLS} == names
    assert "<UNK>" not in {vocab.itos[i] for i in G.special_token_ids(vocab, allow_unk=True)}


def _rigged_lstm(vocab: Vocab) -> MusicLSTM:
    """An LSTM whose output strongly prefers every special token."""
    model = tiny_lstm(len(vocab))
    with torch.no_grad():
        model.fc2.bias.zero_()
        for i in G.special_token_ids(vocab):
            model.fc2.bias[i] = 8.0
    return model


@pytest.mark.parametrize("incremental", [True, False])
def test_generation_never_emits_special_or_style_tokens(incremental: bool) -> None:
    vocab = tiny_vocab()
    model = _rigged_lstm(vocab)
    banned = G.special_token_ids(vocab)
    gen = torch.Generator().manual_seed(3)
    out = G._sample_ids(model, [5, 6, 7, 8], num_tokens=200, seq_len=16,
                        device=torch.device("cpu"), temperature=1.0, top_k=0, top_p=0.0,
                        banned_ids=banned, generator=gen, incremental=incremental)
    assert len(out) == 200
    assert not set(out) & set(banned)


def test_incremental_lstm_matches_full_window_while_context_fits() -> None:
    """Carrying (h, c) must reproduce the full-window path token for token for
    as long as the whole context still fits in seq_len (after that the window
    path starts truncating history and the two legitimately diverge)."""
    vocab_size = 40
    model = tiny_lstm(vocab_size, seed=7)
    seq_len, seed = 48, [5, 9, 11, 13, 17, 19, 23, 29]
    n = seq_len - len(seed) + 1          # the last token is predicted from a full window
    kwargs = dict(num_tokens=n, seq_len=seq_len, device=torch.device("cpu"),
                  temperature=1.0, top_k=0, top_p=0.0)
    full = G._sample_ids(model, seed, generator=torch.Generator().manual_seed(11),
                         incremental=False, **kwargs)
    inc = G._sample_ids(model, seed, generator=torch.Generator().manual_seed(11),
                        incremental=True, **kwargs)
    assert full == inc


def test_incremental_logits_match_full_forward() -> None:
    model = tiny_lstm(30, seed=2)
    x = torch.randint(5, 30, (2, 20))
    with torch.no_grad():
        full = model(x)
        logits, h = model(x[:, :7], return_hidden=True)
        steps = [logits]
        for t in range(7, 20):
            logits, h = model(x[:, t:t + 1], hidden=h)
            steps.append(logits)
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)


def test_batched_generation_matches_one_sample_at_a_time() -> None:
    model = tiny_lstm(40, seed=5)
    seeds = [[5, 6, 7, 8, 9], [10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
    kwargs = dict(num_tokens=30, seq_len=16, device=torch.device("cpu"),
                  temperature=0.9, top_k=0, top_p=0.0, banned_ids=[0, 1, 2, 3])
    batched = G._sample_many(model, seeds, generators=[torch.Generator().manual_seed(50 + i)
                                                      for i in range(3)], **kwargs)
    solo = [G._sample_ids(model, s, generator=torch.Generator().manual_seed(50 + i), **kwargs)
            for i, s in enumerate(seeds)]
    assert batched == solo


def test_transformer_still_uses_the_full_window_path() -> None:
    torch.manual_seed(0)
    model = MusicTransformer(30, embed_dim=16, hidden_dim=32, num_layers=1, num_heads=2,
                             dropout=0.0, max_len=64).eval()
    out = G._sample_ids(model, [5, 6, 7], num_tokens=40, seq_len=16, device=torch.device("cpu"),
                        temperature=0.9, top_k=0, top_p=0.0,
                        generator=torch.Generator().manual_seed(0))
    assert len(out) == 40 and all(0 <= i < 30 for i in out)


def _write_tiny_checkpoint(tmp_path: Path, cfg) -> Path:
    vocab = tiny_vocab()
    model = _rigged_lstm(vocab)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    vocab.save(ckpt_dir / "vocab.json")
    T._save_checkpoint(ckpt_dir / "best.pt", model, cfg, len(vocab), 1, 1.0)
    return ckpt_dir


def _tiny_generate_cfg(tmp_path: Path, **extra):
    return load_config(**{
        "name": "audit_tiny",
        "model.embed_dim": 16, "model.hidden_dim": 32, "model.num_layers": 2,
        "model.seq_len": 16, "encoding.include_duration": False,
        "train.device": "cpu",
        "generate.num_tokens": 60, "generate.num_samples": 3, "generate.seed": 123,
        "generate.output_dir": str(tmp_path / "out"),
        **extra,
    })


def test_generate_end_to_end_suppresses_specials_and_restores_rng(tmp_path, monkeypatch) -> None:
    cfg = _tiny_generate_cfg(tmp_path)
    ckpt_dir = _write_tiny_checkpoint(tmp_path, cfg)
    pool = [[5, 6, 7, 8, 9, 10, 11, 12] * 5, [13, 12, 11, 10, 9] * 6, [5, 7, 9, 11] * 3]
    monkeypatch.setattr(G, "_load_dataset_pool", lambda _cfg: pool)

    torch.manual_seed(999)
    before = torch.get_rng_state()
    paths = G.generate(cfg, checkpoint_dir=ckpt_dir)
    assert torch.equal(torch.get_rng_state(), before)
    assert len(paths) == 3
    specials = {"<PAD>", "<BOS>", "<EOS>", "<UNK>", *STYLE_SYMBOLS}
    texts = []
    for p in paths:
        symbols = p.with_suffix(".txt").read_text(encoding="utf-8").split("\n")
        assert len(symbols) == 60
        assert not specials & set(symbols)
        texts.append(symbols)

    # Same seed -> bit-identical run.
    again = G.generate(cfg, checkpoint_dir=ckpt_dir)
    assert [p.with_suffix(".txt").read_text(encoding="utf-8").split("\n") for p in again] == texts


# --------------------------------------------------------------------------
# train.py
# --------------------------------------------------------------------------


def _train_cfg(tmp_path: Path, **extra):
    return load_config(**{
        "name": "audit_train",
        "model.embed_dim": 16, "model.hidden_dim": 32, "model.num_layers": 2,
        "model.seq_len": 8, "train.device": "cpu", "train.epochs": 3,
        "train.checkpoint_dir": str(tmp_path / "runs"),
        **extra,
    })


def _fake_loaders(monkeypatch, vocab: Vocab, n_train: int = 40, n_val: int = 16) -> None:
    g = torch.Generator().manual_seed(0)
    V = len(vocab)

    def make(n):
        data = torch.randint(5, V, (n, 9), generator=g)
        return DataLoader(TensorDataset(data[:, :-1], data[:, 1:]), batch_size=16)

    loaders = (make(n_train), make(n_val), make(4), vocab)
    monkeypatch.setattr(T, "dataloaders_from_cache", lambda cfg: loaders)


def _history_epochs(run_dir: Path) -> List[int]:
    with open(run_dir / "history.csv", newline="", encoding="utf-8") as fh:
        return [int(r["epoch"]) for r in csv.DictReader(fh)]


def test_last_checkpoint_records_the_true_last_epoch(tmp_path, monkeypatch) -> None:
    """BUG: last.pt was saved with epoch=best_epoch, so resuming from it
    restarted at best_epoch+1 on top of weights from a later epoch."""
    vocab = tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    cfg = _train_cfg(tmp_path, **{"train.epochs": 3, "train.lr": 0.0})  # lr 0: never improves after epoch 1
    T.train(cfg)
    last = torch.load(tmp_path / "runs" / "audit_train" / "last.pt", weights_only=False)
    assert last["last_epoch"] == 3
    assert last["epoch"] == 1        # best.pt semantics kept for the "epoch" key


def test_resume_does_not_duplicate_history_rows(tmp_path, monkeypatch) -> None:
    """BUG: resuming re-ran epochs after best_epoch and appended them again."""
    vocab = tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    T.train(_train_cfg(tmp_path, **{"train.epochs": 2, "train.lr": 0.0}))
    T.train(_train_cfg(tmp_path, **{"train.epochs": 4, "train.lr": 0.0, "train.resume": True}))
    assert _history_epochs(tmp_path / "runs" / "audit_train") == [1, 2, 3, 4]


def test_resume_truncates_history_written_after_the_checkpoint(tmp_path, monkeypatch) -> None:
    vocab = tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    T.train(_train_cfg(tmp_path, **{"train.epochs": 2}))
    run_dir = tmp_path / "runs" / "audit_train"
    with open(run_dir / "history.csv", "a", encoding="utf-8") as fh:
        fh.write("3,1,1,1,0.001,1,0\n")    # a row from an epoch that was never checkpointed
    T.train(_train_cfg(tmp_path, **{"train.epochs": 3, "train.resume": True}))
    assert _history_epochs(run_dir) == [1, 2, 3]


def test_resuming_an_early_stopped_run_does_nothing(tmp_path, monkeypatch) -> None:
    """BUG: patience state was not saved, so resuming a run that had already
    early-stopped silently trained `patience` more epochs."""
    vocab = tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    base = {"train.epochs": 10, "train.patience": 2, "train.lr": 0.0}
    T.train(_train_cfg(tmp_path, **base))
    run_dir = tmp_path / "runs" / "audit_train"
    assert _history_epochs(run_dir) == [1, 2, 3]
    T.train(_train_cfg(tmp_path, **base, **{"train.resume": True}))
    assert _history_epochs(run_dir) == [1, 2, 3]


def test_resume_with_mismatched_shapes_says_why(tmp_path, monkeypatch) -> None:
    vocab = tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    T.train(_train_cfg(tmp_path, **{"train.epochs": 1}))
    with pytest.raises(RuntimeError, match="architecture"):
        T.train(_train_cfg(tmp_path, **{"train.epochs": 2, "train.resume": True,
                                        "model.hidden_dim": 64}))


def test_interrupted_checkpoint_write_leaves_best_intact(tmp_path, monkeypatch) -> None:
    """BUG: torch.save wrote best.pt in place; dying mid-write left a truncated
    best.pt, i.e. destroyed the one file the run exists to produce."""
    vocab = tiny_vocab()
    cfg = _train_cfg(tmp_path)
    model = tiny_lstm(len(vocab))
    path = tmp_path / "best.pt"
    T._save_checkpoint(path, model, cfg, len(vocab), 1, 1.0)
    good = path.read_bytes()

    real_save = torch.save

    def dying_save(obj, f, *a, **k):
        with open(f, "wb") as fh:
            fh.write(b"partial")
        raise OSError("disk yanked")

    monkeypatch.setattr(T.torch, "save", dying_save)
    with pytest.raises(OSError):
        T._save_checkpoint(path, model, cfg, len(vocab), 2, 0.5)
    monkeypatch.setattr(T.torch, "save", real_save)
    assert path.read_bytes() == good
    assert not list(tmp_path.glob("*.tmp*"))


def test_epoch_loss_is_the_exact_per_token_mean_with_padding() -> None:
    torch.manual_seed(0)
    V = 12
    model = tiny_lstm(V)
    x = torch.randint(1, V, (5, 6))
    y = torch.randint(1, V, (5, 6))
    y[0, :3] = 0
    y[4, :] = 0
    loader = DataLoader(TensorDataset(x, y), batch_size=2)   # uneven final batch
    crit = nn.CrossEntropyLoss(ignore_index=0)
    got = T._run_epoch(model, loader, crit, torch.device("cpu"))
    with torch.no_grad():
        ref = nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=0)
    assert math.isclose(got, float(ref), rel_tol=1e-5)


def test_validation_pass_is_deterministic_and_graph_free() -> None:
    V = 12
    model = tiny_lstm(V)
    model.train()                                   # left in train mode by a previous pass
    x = torch.randint(1, V, (6, 6))
    loader = DataLoader(TensorDataset(x, x), batch_size=3)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    a = T._run_epoch(model, loader, crit, torch.device("cpu"))
    b = T._run_epoch(model, loader, crit, torch.device("cpu"))
    assert a == b                                   # dropout off
    assert all(p.grad is None for p in model.parameters())


# --------------------------------------------------------------------------
# evaluate.py
# --------------------------------------------------------------------------


def _scale_piece(tonic: int, steps=(0, 2, 4, 5, 7, 9, 11), reps: int = 4) -> Piece:
    events, t = [], 0.0
    for _ in range(reps):
        for s in steps:
            events.append(NoteEvent(pitch=60 + (tonic + s) % 12, start=t, duration=0.25))
            t += 0.25
        events.append(NoteEvent(pitch=60 + tonic, start=t, duration=1.0))   # land on the tonic
        t += 1.0
    return Piece(events=events)


def test_pc_distance_does_not_reward_atonal_output() -> None:
    """BUG: pc_distance pooled the reference across every key. With +/-6
    transposition that pool is flat, so a uniformly chromatic (atonal) sample
    matched it better than a perfectly tonal one."""
    reference = [_scale_piece(k) for k in range(12)]
    tonal = [_scale_piece(7)]                                   # G major
    atonal = [_scale_piece(0, steps=tuple(range(12)))]          # chromatic
    d_tonal = E.evaluate_pieces(tonal, reference)["pc_distance"]
    d_atonal = E.evaluate_pieces(atonal, reference)["pc_distance"]
    assert d_tonal < d_atonal
    assert d_tonal < 0.05


def test_perplexity_restores_train_mode_and_builds_no_graph() -> None:
    V = 12
    model = tiny_lstm(V).train()
    x = torch.randint(1, V, (4, 6))
    ppl = E.perplexity(model, DataLoader(TensorDataset(x, x), batch_size=2))
    assert math.isfinite(ppl) and ppl > 1
    assert model.training
    assert all(p.grad is None for p in model.parameters())


# --------------------------------------------------------------------------
# config.py / main.py
# --------------------------------------------------------------------------


def test_override_scientific_notation_is_a_float() -> None:
    """BUG: yaml.safe_load('1e-4') is the *string* '1e-4' (YAML 1.1 wants a dot)."""
    from src.config import parse_override_value
    assert parse_override_value("1e-4") == pytest.approx(1e-4)
    assert parse_override_value("256") == 256
    assert parse_override_value("false") is False
    assert parse_override_value("[-3,3]") == [-3, 3]
    assert parse_override_value("adl_run") == "adl_run"


def test_list_override_replaces_the_list() -> None:
    cfg = load_config("configs/adl_mixed.yaml", **{"augment.transpose_range": [-3, 3]})
    assert cfg.augment.transpose_range == [-3, 3]
    assert cfg.augment.pitch_range == [21, 108]          # siblings from default.yaml survive


def test_deep_merge_does_not_alias_the_override() -> None:
    from src.config import _deep_merge
    over = {"a": {"b": [1, 2]}}
    merged = _deep_merge({"a": {"c": 1}}, over)
    merged["a"]["b"].append(3)
    assert over["a"]["b"] == [1, 2]


def test_dotted_override_through_a_scalar_fails_clearly() -> None:
    with pytest.raises(ValueError, match="train.lr"):
        load_config(**{"train.lr.x": 1})


def test_set_on_both_sides_of_the_subcommand_is_merged(monkeypatch) -> None:
    """BUG: `main.py --set a=1 generate --set b=2` dropped a=1 -- the
    subparser's --set list replaced the top-level one."""
    import main as M

    seen = {}

    def fake_generate(config_path, overrides, cfg, **kw):
        seen["overrides"] = list(overrides)

    monkeypatch.setattr(M, "run_generate", fake_generate)
    rc = M.main(["--set", "generate.num_tokens=10", "generate",
                 "--set", "generate.top_k=5", "--config", "configs/smoke.yaml"])
    assert rc == 0
    assert seen["overrides"] == ["generate.num_tokens=10", "generate.top_k=5"]


def test_demo_honours_set_and_force_prepare(monkeypatch) -> None:
    """BUG: `main.py demo --set ... --force-prepare` parsed both flags and then
    ignored them."""
    import main as M

    seen = {}
    monkeypatch.setattr(M, "run_demo", lambda timings, overrides=(), force_prepare=False:
                        seen.update(overrides=list(overrides), force=force_prepare))
    assert M.main(["demo", "--set", "train.epochs=1", "--force-prepare"]) == 0
    assert seen == {"overrides": ["train.epochs=1"], "force": True}


# --------------------------------------------------------------------------
# generate.seed_split -- prime from held-out music
# --------------------------------------------------------------------------


def _fake_processed(monkeypatch) -> None:
    splits = {"train": [[5, 5, 5, 5]], "val": [[6, 6, 6, 6]], "test": [[7, 7, 7, 7]]}
    monkeypatch.setattr(G, "load_split", lambda cfg, name: splits[name])


def test_seed_pool_defaults_to_the_val_split(monkeypatch) -> None:
    """BUG: generation was primed from the TRAIN split. v3 overfit after epoch
    6, so a train primer invites it to continue a piece it memorised."""
    _fake_processed(monkeypatch)
    assert G._load_dataset_pool(load_config()) == [[6, 6, 6, 6]]
    for split, tok in (("train", 5), ("val", 6), ("test", 7)):
        cfg = load_config(**{"generate.seed_split": split})
        assert G._load_dataset_pool(cfg) == [[tok] * 4]


def test_unknown_seed_split_is_rejected(monkeypatch) -> None:
    _fake_processed(monkeypatch)
    with pytest.raises(ValueError, match="seed_split"):
        G._load_dataset_pool(load_config(**{"generate.seed_split": "validation"}))


# --------------------------------------------------------------------------
# evaluate -- memorisation metrics
# --------------------------------------------------------------------------


def test_longest_common_run() -> None:
    assert E.longest_common_run([1, 2, 3, 4, 9], [0, 1, 2, 3, 4, 5]) == 4
    assert E.longest_common_run([1, 2], [3, 4]) == 0
    assert E.longest_common_run([], [1]) == 0


def test_ngram_index_overlap() -> None:
    index = E.NGramIndex([[1, 2, 3, 4, 5, 6]], n=4)
    assert index.overlap([1, 2, 3, 4, 5]) == 1.0        # (1234), (2345) both in train
    assert index.overlap([1, 2, 3, 4, 9]) == 0.5
    assert index.overlap([1, 2, 3]) is None             # too short for one 4-gram


def test_copy_report_flags_a_verbatim_continuation() -> None:
    train = [list(range(10, 60))]
    index = {n: E.NGramIndex(train, n=n) for n in (4, 8)}
    copied = E.copy_metrics(generated=list(range(30, 50)), truth=list(range(30, 50)), indexes=index)
    fresh = E.copy_metrics(generated=[7, 8, 7, 9, 7, 8, 9, 9] * 3, truth=list(range(30, 54)), indexes=index)
    assert copied["lcs_run"] == 20 and copied["train_4gram"] == 1.0 and copied["train_8gram"] == 1.0
    assert fresh["lcs_run"] == 0 and fresh["train_4gram"] == 0.0


def test_fresh_run_restarts_history(tmp_path, monkeypatch) -> None:
    """BUG: a non-resume run in an existing run directory appended to the old
    history.csv (runs/smoke/history.csv read 1 2 3 1 2 3 1 2 3 ...)."""
    vocab = tiny_vocab()
    _fake_loaders(monkeypatch, vocab)
    T.train(_train_cfg(tmp_path, **{"train.epochs": 2}))
    T.train(_train_cfg(tmp_path, **{"train.epochs": 2}))
    assert _history_epochs(tmp_path / "runs" / "audit_train") == [1, 2]
