"""Parametric noise models and coverage checks for the mismatch-counting experiments.

All times are in nanoseconds.

`rz` is a virtual gate on IBM hardware: it is implemented as a frame change in
software, so it takes zero time and carries essentially no error. By default
(`virtual_rz=True`) it therefore receives neither depolarizing error nor
thermal relaxation. Pass `virtual_rz=False` for the older, strictly pessimistic
behaviour in which rz is treated like any other one-qubit gate.
"""
from __future__ import annotations

import itertools

from qiskit_aer.noise import (NoiseModel, ReadoutError, depolarizing_error,
                              thermal_relaxation_error)

BASIS_GATES = ["cx", "rz", "sx", "x"]
ONE_QUBIT_GATES = ("rz", "sx", "x")
TWO_QUBIT_GATES = ("cx",)
VIRTUAL_GATES = ("rz",)


def noisy_one_qubit_gates(virtual_rz=True):
    """The one-qubit gates that carry noise, given the virtual-rz convention."""
    if not virtual_rz:
        return ONE_QUBIT_GATES
    return tuple(gate for gate in ONE_QUBIT_GATES if gate not in VIRTUAL_GATES)


def build_noise_model(num_qubits, p_1q=0.0, p_2q=0.0, t1=None, t2=None,
                      gate_time_1q=50.0, gate_time_2q=300.0,
                      readout_error=None, qubits=None, virtual_rz=True):
    """Depolarizing noise, optionally composed with thermal relaxation and readout error.

    Errors are attached per qubit (and per ordered qubit pair for `cx`) rather
    than globally, so that `qubits` can restrict coverage to a subset.
    Thermal relaxation is enabled only when both `t1` and `t2` are given.
    With `virtual_rz` (the default) rz gets no depolarizing error and no
    thermal relaxation, matching hardware where it is a zero-duration frame
    change rather than a pulse.
    """
    qubits = list(range(num_qubits)) if qubits is None else list(qubits)
    thermal = t1 is not None and t2 is not None
    if thermal and t2 > 2 * t1:
        raise ValueError(f"T2={t2} exceeds 2*T1={2 * t1}")

    noise_model = NoiseModel(basis_gates=BASIS_GATES)

    error_1q = depolarizing_error(p_1q, 1)
    if thermal:
        error_1q = error_1q.compose(thermal_relaxation_error(t1, t2, gate_time_1q))
    for qubit in qubits:
        for gate in noisy_one_qubit_gates(virtual_rz):
            noise_model.add_quantum_error(error_1q, gate, [qubit])

    error_2q = depolarizing_error(p_2q, 2)
    if thermal:
        relax = thermal_relaxation_error(t1, t2, gate_time_2q)
        error_2q = error_2q.compose(relax.tensor(relax))
    for pair in itertools.permutations(qubits, 2):
        for gate in TWO_QUBIT_GATES:
            noise_model.add_quantum_error(error_2q, gate, list(pair))

    if readout_error:
        matrix = [[1 - readout_error, readout_error],
                  [readout_error, 1 - readout_error]]
        for qubit in qubits:
            noise_model.add_readout_error(ReadoutError(matrix), [qubit])

    return noise_model


def covered_operations(noise_model):
    """Map operation name -> set of covered qubit tuples, or None when all qubits are covered."""
    covered = {}
    for entry in noise_model.to_dict()["errors"]:
        gate_qubits = entry.get("gate_qubits")
        for operation in entry["operations"]:
            if gate_qubits is None:
                covered[operation] = None
            elif covered.get(operation, set()) is not None:
                covered.setdefault(operation, set()).update(
                    tuple(q) for q in gate_qubits)
    return covered


def assert_full_coverage(noise_model, circuit, check_measure=False, exempt=(),
                         skip=("barrier", "delay", "reset", "if_else", "store")):
    """Raise unless every gate instance in `circuit` has a matching error in `noise_model`.

    `exempt` names gates that are deliberately noiseless and so must not be
    reported as missing -- pass VIRTUAL_GATES when the model was built with
    `virtual_rz=True`, otherwise every rz in the circuit looks uncovered.

    `if_else` and `store` (from depth_optimal_counter.py's classically
    conditioned rotations) are classical control-flow constructs, not
    physical gates, so they are skipped by default like barrier/reset. NOTE:
    this only checks gates visible in circuit.data's flat iteration -- gates
    INSIDE an if_else block's body (e.g. the conditioned Rz corrections) are
    not iterated here at all, so this function does not verify their
    coverage one way or the other. Whether Aer's noise model actually applies
    to gates inside a conditional block at simulation time is a separate,
    unverified question; see results/README.md.
    """
    covered = covered_operations(noise_model)
    missing = set()
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in skip or name in exempt or (name == "measure" and not check_measure):
            continue
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        entry = covered.get(name, set())
        if entry is not None and qubits not in entry:
            missing.add((name, qubits))
    if missing:
        listed = ", ".join(f"{name}{qubits}" for name, qubits in sorted(missing))
        raise AssertionError(f"noise model does not cover: {listed}")
