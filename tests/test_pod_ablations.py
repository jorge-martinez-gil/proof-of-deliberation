"""Unit tests for the PoD-ablation flag set and the dispatcher hook."""

from __future__ import annotations

from pod.baselines import (
    PoD_ABLATIONS,
    PoDAblation,
    get_pod_ablation,
    is_pod_family,
)


def test_full_pod_enables_all_layers():
    spec = get_pod_ablation("PoD")
    assert spec is not None
    assert spec == PoDAblation(True, True, True)


def test_no_gate_disables_gate_only():
    spec = get_pod_ablation("PoD-NoGate")
    assert spec is not None
    assert spec.use_gate is False
    assert spec.use_coupling is True
    assert spec.use_vigilance is True


def test_no_coupling_disables_coupling_only():
    spec = get_pod_ablation("PoD-NoCoupling")
    assert spec is not None
    assert spec.use_gate is True
    assert spec.use_coupling is False
    assert spec.use_vigilance is True


def test_no_vigilance_disables_vigilance_only():
    spec = get_pod_ablation("PoD-NoVigilance")
    assert spec is not None
    assert spec.use_gate is True
    assert spec.use_coupling is True
    assert spec.use_vigilance is False


def test_is_pod_family_recognises_all_variants():
    for name in ("PoD", "PoD-NoGate", "PoD-NoCoupling", "PoD-NoVigilance"):
        assert is_pod_family(name)


def test_is_pod_family_rejects_non_members():
    assert not is_pod_family("AL")
    assert not is_pod_family("StaticGating")
    assert not is_pod_family("AdaptiveGating")
    assert not is_pod_family("WorkerQuality")
    assert not is_pod_family("PoD-Nonsense")


def test_pod_ablation_table_is_immutable():
    # Frozen dataclass -> attribute assignment must fail.
    spec = PoD_ABLATIONS["PoD"]
    import dataclasses
    assert dataclasses.is_dataclass(spec)
    try:
        spec.use_gate = False  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("PoDAblation should be frozen")
