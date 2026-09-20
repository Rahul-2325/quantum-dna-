import numpy as np, itertools, warnings
warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector

def weight_unitary(n):
    """Paper-2 Algorithm 2 (no zero padding), UNITARY form (no measurement):
       |x>_S |0>_C  ->  |x>_S |W(x)>_C ,  C = k = ceil(log2(n+1)) qubits (little-endian)."""
    k = int(np.ceil(np.log2(n + 1))); N = 2**k - 1
    qc = QuantumCircuit(k + n, name=f"HW{n}")
    qc.h(range(k))
    for j in range(k):
        qc.rz(np.pi * n / (N + 1) * 2**j, j)                    # gamma_j
        for q in range(k, k + n):
            qc.crz(2 * np.pi / (N + 1) * 2**j, j, q)            # beta_j
    qc.append(QFTGate(k).inverse(), range(k))
    return qc, k

def _cswap_reverse(qc, ctrl, wires):
    for i in range(len(wires) // 2):
        qc.cswap(ctrl, wires[i], wires[len(wires) - 1 - i])

def controlled_rotate_right(qc, ctrl, wires, s):
    """new[i] = old[(i-s) mod n], controlled on `ctrl`, via 3 reversals (~n Fredkins)."""
    n = len(wires); s %= n
    if s == 0: return
    _cswap_reverse(qc, ctrl, wires)
    _cswap_reverse(qc, ctrl, wires[:s])
    _cswap_reverse(qc, ctrl, wires[s:])

def hwb_circuit(n):
    """Quantum hwb: cyclic shift right by Hamming weight, using k=ceil(log2(n+1)) ancillas
       (compute weight -> controlled shifts by 2^j -> uncompute)."""
    W, k = weight_unitary(n)
    qc = QuantumCircuit(k + n, name=f"hwb{n}")
    qc.compose(W, inplace=True)
    data = list(range(k, k + n))
    for j in range(k):
        controlled_rotate_right(qc, j, data, 2**j)
    qc.compose(W.inverse(), inplace=True)
    return qc, k

def hwb_classical(bits):                      # bits = tuple (x1..xn); shift right by weight
    n, w = len(bits), sum(bits)
    return tuple(bits[(i - w) % n] for i in range(n))

# ---- DNA mismatch counting -------------------------------------------------
ENC = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}

def mismatch_circuit(read, ref_window):
    """Counts base mismatches between `read` and a same-length reference window.
       qubits: [control k | flags L | read 2L].  Read loaded by X gates; the reference window is
       applied as X gates (read <- read XOR ref); flag_i = OR of the two XOR bits;
       then Paper-2 Alg.2 weight computation on the flag register."""
    L = len(read); W, k = weight_unitary(L)
    qc = QuantumCircuit(k + L + 2 * L)
    flags = list(range(k, k + L)); rd = list(range(k + L, k + 3 * L))
    for i, b in enumerate(read):
        for t, bit in enumerate(ENC[b]):
            if bit: qc.x(rd[2 * i + t])
    for i, b in enumerate(ref_window):                       # compare: read <- read XOR ref
        for t, bit in enumerate(ENC[b]):
            if bit: qc.x(rd[2 * i + t])
    for i in range(L):                                       # flag = a OR b
        a, b = rd[2 * i], rd[2 * i + 1]
        qc.x(a); qc.x(b); qc.ccx(a, b, flags[i]); qc.x(flags[i]); qc.x(a); qc.x(b)
    # weight computation acts on control [0..k-1] + flags[k..k+L-1]
    qc.compose(W, qubits=list(range(k + L)), inplace=True)
    return qc, k

def read_control_value(qc, k):
    d = Statevector(qc).probabilities_dict(qargs=list(range(k)))
    best = max(d, key=d.get)
    return int(best, 2), d[best]
