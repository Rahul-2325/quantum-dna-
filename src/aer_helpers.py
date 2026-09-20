"""Outcome-distribution helpers for running hwlib circuits on Aer."""
from __future__ import annotations

from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from hwlib import mismatch_circuit
from noise_models import BASIS_GATES


def control_distribution_statevector(qc, k):
    """Full {value: probability} distribution of the k-qubit control register.

    `qc` must carry no measurements (Statevector rejects them).
    """
    probabilities = Statevector(qc).probabilities_dict(qargs=list(range(k)))
    return {int(bits, 2): p for bits, p in probabilities.items()}


def control_distribution_aer(qc, shots, noise_model=None, seed=None,
                             method="matrix_product_state"):
    """Full {value: probability} distribution over `qc`'s measured register, from Aer shots.

    `qc` must already measure the control qubits, and must already be
    transpiled to BASIS_GATES whenever a noise model is supplied.

    The default MPS method is exact for these low-entanglement circuits (the
    read register stays in a product basis state) and is ~23x faster than dense
    statevector at L=6; test_mps_matches_statevector pins that equivalence.
    """
    simulator = AerSimulator(noise_model=noise_model, method=method)
    counts = simulator.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    return {int(bits, 2): count / shots for bits, count in counts.items()}


def mismatch_circuit_measured(read, ref_window):
    """mismatch_circuit with the k control qubits measured into a classical register."""
    qc, k = mismatch_circuit(read, ref_window)
    creg = ClassicalRegister(k, "c")
    qc.add_register(creg)
    qc.measure(list(range(k)), list(creg))
    return qc, k


def transpile_for_noise(qc, seed=1234, optimization_level=1):
    """Transpile to BASIS_GATES with all-to-all connectivity (no coupling map)."""
    return transpile(qc, basis_gates=BASIS_GATES, coupling_map=None,
                     optimization_level=optimization_level, seed_transpiler=seed)
