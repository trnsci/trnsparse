"""CPU tests for the static kernels' work split across physical cores (#30)."""

from __future__ import annotations

import pytest

from trnsparse.nki import kernels as K


@pytest.mark.parametrize("n_prog", [1, 2, 3])
@pytest.mark.parametrize(
    "costs",
    [
        [],
        [1.0],
        [3, 3, 3, 3],
        [41, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # global-token row first
        [0.25, 5, 0.25, 7, 2, 2, 9],
    ],
)
def test_assign_units_partitions(costs, n_prog):
    owned = [K._assign_units(costs, n_prog, pid) for pid in range(n_prog)]
    flat = sorted(u for o in owned for u in o)
    assert flat == list(range(len(costs)))  # every unit exactly once: no shared writes
    for o in owned:
        assert o == sorted(o)


def test_assign_units_balances_skew():
    costs = [41, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    loads = [sum(costs[u] for u in K._assign_units(costs, 2, pid)) for pid in range(2)]
    assert max(loads) == 41  # the heavy row alone; the rest go to the other core
    even = [4] * 10
    loads = [sum(even[u] for u in K._assign_units(even, 2, pid)) for pid in range(2)]
    assert loads == [20, 20]


def test_assign_units_deterministic():
    costs = [5, 1, 5, 1, 2, 2]
    assert K._assign_units(costs, 2, 0) == K._assign_units(list(costs), 2, 0)


def test_core_split_env(monkeypatch):
    monkeypatch.setenv("TRNSPARSE_CORE_SPLIT", "0")
    assert not K._core_split_enabled()
    monkeypatch.setenv("TRNSPARSE_CORE_SPLIT", "1")
    assert K._core_split_enabled()
    monkeypatch.delenv("TRNSPARSE_CORE_SPLIT")
    assert K._core_split_enabled()


def test_row_block_counts():
    assert K._row_block_counts((0, 2, 2, 5)) == [2, 0, 3]
