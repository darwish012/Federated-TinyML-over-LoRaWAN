"""Study geometry and the machine order (amendment2 machine_order_v2)."""
from __future__ import annotations

import pytest

import study_plan as sp


@pytest.fixture(scope="module")
def research_available():
    try:
        sp.load_fl_modules()
    except FileNotFoundError:
        pytest.skip("research folder not available")


def test_levels_validation():
    assert sp.validate_levels([4, 8, 28]) == [4, 8, 28]
    for bad in ([], [8, 4], [4, 4], [5], [32]):
        with pytest.raises(ValueError):
            sp.validate_levels(bad)


def test_active_positions_boards_and_positions_per_board():
    assert sp.active_positions(4) == (0, 1, 2, 3)
    assert sp.boards_for_level(4) == (1, 2, 3, 4)
    assert sp.boards_for_level(12) == tuple(range(1, 11))
    # Level 28: boards 1..8 host three machines, boards 9 and 10 host two.
    for board in range(1, 9):
        assert len(sp.positions_on_board(board, 28)) == 3
    assert sp.positions_on_board(9, 28) == (8, 18)
    assert sp.positions_on_board(10, 28) == (9, 19)
    assert sp.positions_on_board(1, 28) == (0, 10, 20)
    assert sp.max_positions_per_board(28) == 3
    assert sp.max_positions_per_board(4) == 1


def test_standalone_order_equals_the_research_module(research_available):
    amendment = sp.load_amendment2()
    standalone = sp.build_order_standalone(amendment)
    from_module = sp.order_from_research_module()
    assert standalone == from_module
    assert len(standalone) == 54 and len(set(standalone)) == 54


def test_order_puts_the_41_eligible_machines_first(research_available):
    amendment = sp.load_amendment2()
    order, source = sp.load_machine_order()
    assert source == "machine_order_v2"
    assert set(order[:41]) == set(amendment["hardware_eligibility"]["eligible_41"])
    assert set(order[41:]) == set(amendment["hardware_eligibility"]["ineligible_13"])
    # Every hardware level (<= 28 machines) draws only eligible machines.
    assert set(order[:28]) <= set(amendment["hardware_eligibility"]["eligible_41"])


def test_order_is_nested_and_deterministic(research_available):
    first, _ = sp.load_machine_order()
    second = sp.build_order_standalone(sp.load_amendment2())
    assert first[:4] == second[:4]
    assert first == second
    # datasets are interleaved round-robin: the first machines span several datasets
    assert len({m.split("::")[0] for m in first[:9]}) >= 6


def test_fl_modules_are_the_research_ones(research_available):
    twin_server, fl = sp.load_fl_modules()
    assert twin_server.__file__.endswith("twin_server.py")
    assert "hardware_twin_scaling_20260929" in twin_server.__file__
    assert "fl_algorithm_comparison_20260929" in fl.__file__
    assert (twin_server.ETA, twin_server.B1, twin_server.B2, twin_server.TAU) == (
        0.01, 0.9, 0.99, 0.001)
