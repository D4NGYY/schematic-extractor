from __future__ import annotations

from src.core.scope import assess_scope


def test_bus_dominated_out_of_scope() -> None:
    a = assess_scope(num_components=8, num_nets=517)  # rams
    assert a.in_scope is False
    assert a.confidence == "low"
    assert "bus-dominated" in a.reason


def test_too_large_out_of_scope() -> None:
    a = assess_scope(num_components=97, num_nets=143)  # electric
    assert a.in_scope is False
    assert "too large" in a.reason


def test_normal_board_in_scope() -> None:
    a = assess_scope(num_components=8, num_nets=11)  # sallen_key
    assert a.in_scope is True
    assert a.confidence == "high"
    assert a.nets_per_component == 1.38


def test_zero_components_not_in_scope() -> None:
    a = assess_scope(num_components=0, num_nets=0)
    assert a.in_scope is False  # inf ratio


def test_thresholds_configurable() -> None:
    # carte_test (42, 261, r6.2) is in-scope at default 8 but out at stricter 6
    assert assess_scope(42, 261).in_scope is True
    assert assess_scope(42, 261, max_nets_per_component=6.0).in_scope is False
