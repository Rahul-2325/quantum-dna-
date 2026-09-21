"""Task 3, combined piece: post-selection stacked with ZNE.

Post-selection and ZNE were tested separately (post_selection.py, zne.py).
This stacks them: at EACH noise scale (1, 3, 5, ...), discard shots landing
on an impossible (out-of-range) outcome BEFORE computing P(correct), then
extrapolate the resulting post-selected series to scale=0 -- rather than
extrapolating the raw (non-post-selected) series, which is what zne.py's
probe did on its own.

The two techniques should compose (post-selection is a free, per-scale
filter; ZNE reasons about the remaining, filtered trend), but whether
stacking them actually beats either one alone -- rather than, say, the folded
circuits' higher out-of-range rate eating into the sample used for
extrapolation -- is an empirical question this script answers directly
rather than assumes.
"""
from __future__ import annotations

from aer_helpers import control_distributions_aer, transpile_for_noise
from experiments.noise_degradation import scenario_noise_model
from hwlib import mismatch_circuit
from zne import exponential_extrapolate, fold_and_measure, linear_extrapolate

SCALES = (1, 3, 5)


def distribution_at_scale(read, window, L, p, scale, shots, seed):
    qc, k = mismatch_circuit(read, window)
    folded = fold_and_measure(qc, k, scale)
    transpiled = transpile_for_noise(folded)
    noise_model = scenario_noise_model("S2", p, transpiled.num_qubits)
    return control_distributions_aer([transpiled], shots, noise_model=noise_model,
                                     seed=seed, num_bits=k)[0]


def raw_and_post_selected(distribution, L, m):
    p_correct = distribution.get(m, 0.0)
    p_oor = sum(prob for value, prob in distribution.items() if value > L)
    kept = 1 - p_oor
    post_selected = p_correct / kept if kept > 0 else float("nan")
    return p_correct, post_selected, p_oor


def combined_estimate(read, window, L, m, p, shots=2048, scales=SCALES, seed_base=100):
    """Returns (raw_scale1, zne_on_raw, zne_on_post_selected, discard_rates)."""
    raw_series, post_series, discards = [], [], []
    for scale in scales:
        distribution = distribution_at_scale(read, window, L, p, scale, shots,
                                             seed=seed_base + scale)
        raw, post, discard = raw_and_post_selected(distribution, L, m)
        raw_series.append(raw)
        post_series.append(post)
        discards.append(discard)

    zne_raw = exponential_extrapolate(scales, raw_series)
    zne_post = exponential_extrapolate(scales, post_series)
    return {"raw_series": raw_series, "post_series": post_series,
            "discard_rates": discards, "raw_scale1": raw_series[0],
            "zne_on_raw": zne_raw, "zne_on_post_selected": zne_post}
