"""Quantum Iterated Greedy (Q-IG) for the permutation flow shop (Cmax) and its classical twins.

Quantum subroutines are emulated exactly in the query model: the makespans of all branches are computed classically
(not charged), and the measurement outcome is drawn from the exact output law of the amplitude-amplification circuit.
"""
__version__ = "1.0.0"
