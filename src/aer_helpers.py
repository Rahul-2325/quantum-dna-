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


MAX_CIRCUITS_PER_JOB = 4
"""Aer's C++ backend has twice now raised a hard, memory-pressure-dependent
`MemoryError: bad allocation` (once mid-L=8-sweep earlier in this project,
once mid-connectivity-aware-sweep on a heavy-hex-routed depth-optimal
circuit at L=6) submitting a full 8-circuit batch in one job. Retrying the
IDENTICAL batch afterward sometimes succeeds -- this is not a deterministic
bug in any circuit, it is peak memory during the batch exceeding whatever
happened to be free at that moment. Capping the batch size directly lowers
that peak, at the modest cost of a few more ~0.2s job-startup overheads.
Chosen empirically: batches of 1, 2 and 4 all ran the crashing case cleanly;
only 8 failed (non-deterministically) -- see the commit introducing this."""


def control_distributions_aer(circuits, shots, noise_model=None, seed=None,
                              method="matrix_product_state", num_bits=None):
    """Distributions for several circuits, submitted in Aer jobs of at most
    MAX_CIRCUITS_PER_JOB circuits each (not necessarily all in one job -- see
    MAX_CIRCUITS_PER_JOB's own docstring for why).

    Building the simulator (and attaching a per-qubit noise model) costs roughly
    0.2s, so batching several circuits into one job rather than one job each is
    still a real saving over the fully-serial alternative; Aer also
    parallelises across the circuits within each job.

    `num_bits`: circuits with a SINGLE classical register (mismatch_circuit,
    mismatch_circuit_adder) return a plain binary string, parsed as-is. A
    circuit with more than one classical register (mismatch_circuit_depth_optimal,
    whose "out" and scratch "temp" registers are separate) returns a
    space-separated key instead, e.g. "10 1010", which plain int(bits, 2)
    cannot parse. Pass `num_bits` (the width of the register that actually
    holds the answer) to take just the trailing `num_bits` characters after
    stripping the space, discarding everything else -- this works whether or
    not the key has a space, so existing single-register callers are
    unaffected by leaving it unset.
    """
    circuits = list(circuits)
    simulator = AerSimulator(noise_model=noise_model, method=method)
    distributions = []
    for start in range(0, len(circuits), MAX_CIRCUITS_PER_JOB):
        chunk = circuits[start:start + MAX_CIRCUITS_PER_JOB]
        # Vary the seed per chunk so chunking doesn't make every sub-batch
        # replay identical shot outcomes when seed is fixed by the caller.
        chunk_seed = None if seed is None else seed + start
        result = simulator.run(chunk, shots=shots, seed_simulator=chunk_seed).result()
        for index in range(len(chunk)):
            distribution = {}
            for bits, count in result.get_counts(index).items():
                key = bits.replace(" ", "")
                value = int(key[-num_bits:] if num_bits else key, 2)
                distribution[value] = distribution.get(value, 0) + count / shots
            distributions.append(distribution)
    return distributions


def control_distribution_aer(qc, shots, noise_model=None, seed=None,
                             method="matrix_product_state", num_bits=None):
    """Full {value: probability} distribution over `qc`'s measured register, from Aer shots.

    `qc` must already measure the control qubits, and must already be
    transpiled to BASIS_GATES whenever a noise model is supplied. See
    control_distributions_aer for `num_bits` (needed for circuits with more
    than one classical register).

    The default MPS method is exact for these low-entanglement circuits (the
    read register stays in a product basis state) and is ~23x faster than dense
    statevector at L=6; test_mps_matches_statevector pins that equivalence.
    """
    return control_distributions_aer([qc], shots, noise_model=noise_model,
                                     seed=seed, method=method, num_bits=num_bits)[0]


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
