"""Arena registry + per-model KL-floor override + per-arena settle isolation.

Arenas make "try a new model" a config row (its sglang pin, image, seam subset, KL
floors, engine kwargs) without disturbing the validated default path. Pin: the default
arena equals pre-arena behavior; a non-default arena's KL floor overrides the slot
default; and scores from different arenas don't share a championship bracket.
"""

from optima.arenas import DEFAULT_ARENA, Arena, arena_for_model, get_arena, list_arenas
from optima.commit_reveal import Ledger, make_commitment


def test_default_arena_equals_pre_arena_pin():
    # PINNED_SGLANG must alias the default arena's version (back-compat for importers).
    from optima.compat import PINNED_SGLANG
    assert DEFAULT_ARENA.sglang_version == PINNED_SGLANG
    assert get_arena(None) is DEFAULT_ARENA
    assert get_arena("default") is DEFAULT_ARENA


def test_unknown_arena_raises_with_known_list():
    import pytest
    with pytest.raises(KeyError, match="unknown arena"):
        get_arena("does-not-exist")


def test_seam_subset_semantics():
    full = Arena(name="f", model_path="m", sglang_version="x")  # empty subset = all
    assert full.applies_seam("moe") and full.applies_seam("collective")
    subset = Arena(name="s", model_path="m", sglang_version="x", seam_adapters=("attention", "moe"))
    assert subset.applies_seam("attention") and subset.applies_seam("moe")
    assert not subset.applies_seam("collective")


def test_kl_floor_override_and_competable():
    a = Arena(name="m3", model_path="MiniMax/M3", sglang_version="0.5.13",
              kl_floors={"attention.decode": 0.04})
    assert a.kl_floor_for("attention.decode") == 0.04
    assert a.kl_floor_for("norm.rmsnorm") is None  # falls back to slot/CLI
    assert a.competable()
    stub = Arena(name="stub", model_path="x", sglang_version="")  # declared, no pin yet
    assert not stub.competable()


def test_compat_runs_per_arena_seam_subset():
    # A subset arena's canary only iterates its seams (table-driven loop). We assert the
    # subset is honored by checking the labels mention only the subset's chokepoints.
    from optima.compat import run_checks
    subset = Arena(name="attn-only", model_path="m", sglang_version="z", seam_adapters=("attention",))
    checks = run_checks(subset)
    names = " ".join(c.name for c in checks)
    # sglang isn't installed in CI -> import fails fast; but the arena label must appear.
    assert "attn-only" in names
    # When sglang IS importable, the table loop would skip non-attention seams; we at least
    # confirm no 'seam table: moe' check is emitted for an attention-only arena.
    assert "seam table: moe" not in names


def _score(led, hotkey, ch, slot, score, arena, *, rnd=0, pin="0.5.12.post1"):
    led.commit(hotkey, make_commitment(ch, hotkey, "s"), rnd)
    led.reveal(hotkey, ch, "s", rnd, fingerprint=ch)
    led.record_score(hotkey, ch, rnd, score, kl_mean=0.0, passed=True, sglang_version=pin,
                     slot=slot, arena=arena)


def test_settle_arena_filter_isolates_brackets():
    led = Ledger()
    # Same slot, two arenas: a big speedup on model B must NOT beat model A's champion.
    _score(led, "alice", "H_A", "moe.fused_experts", 1.10, "gpt-oss")
    _score(led, "bob", "H_B", "moe.fused_experts", 1.90, "minimax-m3")
    res_a = led.settle(0, margin=0.02, arena="gpt-oss")
    assert res_a.champion.hotkey == "alice"  # bob's 1.90 (different arena) is excluded
    assert res_a.challenger_score == 1.10


def test_settle_no_arena_filter_is_backward_compatible():
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.30, "default")
    res = led.settle(0, margin=0.02)  # arena=None -> all scores, pre-arena behavior
    assert res.champion.hotkey == "alice"
    assert res.weights == {"alice": 1.0}


def test_arena_score_persists(tmp_path):
    led = Ledger()
    _score(led, "alice", "H_A", "moe.fused_experts", 1.20, "minimax-m3")
    p = tmp_path / "l.json"
    led.save(p)
    led2 = Ledger.load(p)
    assert led2.scores[0].arena == "minimax-m3"


def test_arena_for_model_resolves_then_defaults():
    assert arena_for_model("no-such-model") is DEFAULT_ARENA
    assert "default" in list_arenas()
