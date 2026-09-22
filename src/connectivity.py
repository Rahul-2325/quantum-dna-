"""Connectivity-aware transpilation: a realistic heavy-hex coupling map
instead of the all-to-all assumption every noise result in this project has
used so far (see "All-to-all connectivity" in results/README.md's
Limitations). Generated natively by Qiskit (CouplingMap.from_heavy_hex) --
no IBM account, no hardware access, no external data needed.

IMPORTANT bug this module fixes, not just documents: passing the FULL
heavy-hex map straight to `transpile(coupling_map=...)` makes the transpiled
circuit as wide as the WHOLE map (57 qubits for distance=5), not just the
qubits the logical circuit actually needs -- a first attempt at this did
exactly that and it silently ballooned every circuit to 57 qubits, which
blew up both MPS (>1GB RAM, still climbing, had to be killed) and dense
statevector (2^57 is uncomputable) alike. `subgraph_for` extracts a
connected subset of exactly the qubits needed (via BFS from qubit 0) and
`CouplingMap.reduce`s onto just those, so the transpiled circuit stays the
size it should be while still only using the heavy-hex adjacency.
"""
from __future__ import annotations

import collections

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from noise_models import BASIS_GATES

HEAVY_HEX_DISTANCE = 5
_FULL_MAP = None


def full_heavy_hex_map():
    """Cached CouplingMap.from_heavy_hex(5) -- 57 qubits, 128 edges."""
    global _FULL_MAP
    if _FULL_MAP is None:
        _FULL_MAP = CouplingMap.from_heavy_hex(HEAVY_HEX_DISTANCE)
    return _FULL_MAP


def subgraph_for(num_qubits, seed_qubit=0):
    """A connected `num_qubits`-qubit subset of the heavy-hex map, as its own
    CouplingMap (relabeled 0..num_qubits-1) -- NOT the full 57-qubit map.
    """
    full = full_heavy_hex_map()
    if num_qubits > full.size():
        raise ValueError(f"{num_qubits} qubits exceeds the {full.size()}-qubit "
                         f"heavy-hex map; use a larger HEAVY_HEX_DISTANCE")
    visited = [seed_qubit]
    frontier = collections.deque([seed_qubit])
    while frontier and len(visited) < num_qubits:
        node = frontier.popleft()
        for neighbor in full.neighbors(node):
            if neighbor not in visited:
                visited.append(neighbor)
                frontier.append(neighbor)
                if len(visited) >= num_qubits:
                    break
    return full.reduce(visited)


def transpile_connectivity_aware(qc, seed=1234, optimization_level=1):
    """Like aer_helpers.transpile_for_noise, but routes onto a
    right-sized (not the full 57-qubit) heavy-hex subgraph via SABRE
    layout/routing, instead of assuming all-to-all connectivity.
    """
    coupling_map = subgraph_for(qc.num_qubits)
    return transpile(qc, basis_gates=BASIS_GATES, coupling_map=coupling_map,
                     optimization_level=optimization_level, seed_transpiler=seed)
