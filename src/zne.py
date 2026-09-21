"""Task 3, third piece: zero-noise extrapolation (ZNE) on P(outcome=w).

Global unitary folding: U -> U (U^dagger U)^m, for odd noise-scale factors
lambda = 2m+1 (m=0,1,2,...). Each fold is logically the identity
(U^dagger U = I) but multiplies the circuit's gate count and hence its noise
exposure by lambda, since every gate now really executes `lambda` times.
This is the simplest standard ZNE technique -- global folding rather than
per-gate/local folding, chosen because it needs no per-gate noise-scaling
logic and composes cleanly with circuits already built elsewhere in this
project (mismatch_circuit, mismatch_circuit_adder).

P(correct) is measured at several scale factors under the SAME noise model,
then linearly extrapolated back to lambda=0 (the noiseless limit). This
project already has ground truth for that limit -- S1 at p=0 gives EXACTLY
1.0 -- so the extrapolated estimate can be checked against a known target,
not just eyeballed for plausibility.
"""
from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister


def fold_circuit(qc, scale):
    """U -> U (U^dagger U)^m for odd integer scale = 2m+1. scale=1 is `qc` itself."""
    if scale % 2 != 1 or scale < 1:
        raise ValueError(f"scale must be a positive odd integer, got {scale}")
    m = (scale - 1) // 2
    folded = qc.copy()
    for _ in range(m):
        folded.compose(qc.inverse(), inplace=True)
        folded.compose(qc, inplace=True)
    return folded


def fold_and_measure(unmeasured_qc, k, scale):
    """Fold the UNMEASURED circuit (no classical register yet), then measure
    the k control qubits once, at the end -- folding must happen before
    measurement, since folding a measured circuit would fold the
    measurement too and break the classical-bit accounting."""
    folded = fold_circuit(unmeasured_qc, scale)
    creg = ClassicalRegister(k, "c")
    folded.add_register(creg)
    folded.measure(list(range(k)), list(creg))
    return folded


def linear_extrapolate(scales, values):
    """Fit values ~ a + b*scale by least squares; return the estimate at scale=0."""
    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)
    b, a = np.polyfit(scales, values, 1)
    return a  # intercept = the scale=0 estimate


def exponential_extrapolate(scales, values, floor=1e-6):
    """Fit values ~ a + b*exp(-c*scale) (c>0) by nonlinear least squares
    (scipy.optimize.curve_fit, the standard approach -- e.g. Mitiq's
    ExpFactory -- rather than a hand-derived closed-form formula, which is
    easy to get subtly wrong), then return the estimate at scale=0 (= a+b).

    Falls back to `linear_extrapolate` if the fit doesn't converge (flat or
    non-monotonic data, or too few points for 3 free parameters).
    """
    from scipy.optimize import curve_fit

    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(scales) < 3:
        return linear_extrapolate(scales, values)

    def model(s, a, b, c):
        return a + b * np.exp(-c * s)

    try:
        (a, b, c), _ = curve_fit(
            model, scales, values,
            p0=[values[-1], values[0] - values[-1], 1.0 / max(scales)],
            bounds=([-np.inf, -np.inf, floor], [np.inf, np.inf, np.inf]),
            maxfev=5000)
    except RuntimeError:
        return linear_extrapolate(scales, values)
    return a + b
