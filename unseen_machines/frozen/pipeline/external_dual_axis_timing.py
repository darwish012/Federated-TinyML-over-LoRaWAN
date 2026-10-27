"""Detector-blind dual-axis timing evaluation for external lifecycles.

Eligibility depends only on the frozen analytic onset and lifecycle length.
It never depends on an alarm.  The original V60 single-category result remains
reproducible; this module adds a non-lossy secondary evaluation.
"""

from __future__ import annotations

import math
from typing import Any


def evaluate(first_alarm: int, onset: int | None, n: int) -> dict[str, Any]:
    if int(n) <= 0:
        raise ValueError("n must be positive")
    first_alarm = int(first_alarm)
    if first_alarm >= int(n):
        raise ValueError("first_alarm must be -1 or a zero-based index below n")
    if onset is not None and not (0 <= int(onset) < int(n)):
        raise ValueError("onset must be None or a zero-based index below n")

    terminal = int(math.floor(0.90 * int(n)))
    if onset is None:
        degradation = "onset_unavailable"
        eligibility = "onset_unavailable"
    elif first_alarm < 0:
        degradation = "missed"
        eligibility = (
            "no_feasible_warning_window"
            if int(onset) >= terminal
            else "feasible_warning_window"
        )
    else:
        degradation = (
            "premature" if first_alarm < int(onset) else "onset_detected"
        )
        eligibility = (
            "no_feasible_warning_window"
            if int(onset) >= terminal
            else "feasible_warning_window"
        )

    if first_alarm < 0:
        maintenance = "no_warning"
    elif first_alarm < terminal:
        maintenance = "preterminal_alarm"
    else:
        maintenance = "terminal_band"

    if eligibility == "onset_unavailable":
        predictive = "onset_unavailable"
    elif eligibility == "no_feasible_warning_window":
        predictive = "not_evaluable_empty_window"
    elif first_alarm < 0:
        predictive = "missed"
    elif first_alarm < int(onset):
        predictive = "premature"
    elif first_alarm < terminal:
        predictive = "valid"
    else:
        predictive = "late"

    return {
        "n": int(n),
        "first_alarm": first_alarm,
        "analytic_onset": None if onset is None else int(onset),
        "terminal_boundary": terminal,
        "eligibility": eligibility,
        "degradation_axis": degradation,
        "maintenance_axis": maintenance,
        "predictive_timing": predictive,
        "alarm_minus_onset": (
            None
            if onset is None or first_alarm < 0
            else int(first_alarm - int(onset))
        ),
        "lead_acquisitions_to_last_record": (
            None if first_alarm < 0 else int(n - 1 - first_alarm)
        ),
    }


__all__ = ["evaluate"]
