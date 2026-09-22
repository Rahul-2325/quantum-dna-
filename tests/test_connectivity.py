import pathlib
import sys

import pytest
from qiskit_aer import AerSimulator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aer_helpers import mismatch_circuit_measured
from connectivity import full_heavy_hex_map, subgraph_for, transpile_connectivity_aware
from mismatch_adder import mismatch_circuit_adder_measured


@pytest.mark.parametrize("n", [4, 8, 16, 24, 32])
def test_subgraph_for_is_connected_and_right_sized(n):
    """Regression guard for the bug this module fixes: an earlier version
    handed transpile() the FULL 57-qubit map directly, which silently
    ballooned every transpiled circuit to 57 qubits regardless of how many
    it actually needed, and blew up simulation (>1GB RAM on a circuit that
    should need kilobytes). subgraph_for must always return exactly `n`
    qubits, connected, carved out of the full map -- never the full map
    itself unless n happens to equal its size.
    """
    sub = subgraph_for(n)
    assert sub.size() == n
    assert sub.is_connected()
    assert n < full_heavy_hex_map().size()   # sanity: we're testing real subsets


def test_subgraph_for_rejects_too_large_a_request():
    with pytest.raises(ValueError):
        subgraph_for(full_heavy_hex_map().size() + 1)


@pytest.mark.parametrize("builder", [mismatch_circuit_measured, mismatch_circuit_adder_measured])
def test_connectivity_aware_transpile_preserves_qubit_count(builder):
    """The transpiled circuit must stay the size the LOGICAL circuit needs,
    not balloon to the full coupling map -- this is the exact bug fixed."""
    qc, k = builder("ACGT", "AGGT")
    hexed = transpile_connectivity_aware(qc)
    assert hexed.num_qubits == qc.num_qubits


@pytest.mark.parametrize("builder", [mismatch_circuit_measured, mismatch_circuit_adder_measured])
def test_connectivity_aware_routing_preserves_the_answer(builder):
    """Routing changes GATE PLACEMENT (adds SWAPs), never the logical
    function -- the noisy-free circuit must still give the correct mismatch
    count with near-certainty, exactly like the all-to-all version."""
    read, window = "ACGT", "AGGA"          # 2 mismatches
    qc, k = builder(read, window)
    hexed = transpile_connectivity_aware(qc)

    simulator = AerSimulator(method="matrix_product_state")
    counts = simulator.run(hexed, shots=512, seed_simulator=3).result().get_counts()
    total = sum(counts.values())
    correct = sum(c for key, c in counts.items() if int(key.replace(" ", "")[-k:], 2) == 2)
    assert correct / total > 0.999


def test_connectivity_aware_transpile_adds_real_routing_overhead():
    """Sanity check the whole point of this module: routing onto a
    constrained topology should cost MORE gates than all-to-all, not the
    same (which would mean the coupling map wasn't actually constraining
    anything)."""
    from aer_helpers import transpile_for_noise

    qc, k = mismatch_circuit_measured("ACGTAC", "AGGTAC")   # L=6
    flat = transpile_for_noise(qc)
    hexed = transpile_connectivity_aware(qc)
    assert hexed.count_ops().get("cx", 0) > flat.count_ops().get("cx", 0)
