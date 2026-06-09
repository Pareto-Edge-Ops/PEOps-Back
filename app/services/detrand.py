"""Deterministic helpers shared with the scene builders.

Only `js_round` lives here — it is REAL scene-parity logic (the backend must
format numbers byte-identically to the SPA's JS). The old LCG/seed generators
that fabricated demo data are gone.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal


def js_round(x: float, digits: int = 0) -> float:
    """ES `Number.toFixed` semantics: round the EXACT binary value to the
    nearest decimal; ties pick the larger candidate (toward +Infinity).
    Python's built-in round() is banker's rounding and diverges on exact
    midpoints (0.125 → 0.12 vs JS 0.13). decimal has no HALF_CEILING mode,
    so implement it via floor + half comparison on exact Decimals."""
    quantum = Decimal(1).scaleb(-digits)
    exact = Decimal(x)
    floored = exact.quantize(quantum, rounding=ROUND_FLOOR)
    return float(floored + quantum if exact - floored >= quantum / 2 else floored)
