import numpy as np


def get_fidelity(psi_actual, psi_target, dx):
    """Calculates the quantum fidelity |<target|actual>|^2."""
    overlap = np.sum(np.conj(psi_target) * psi_actual) * dx
    return np.abs(overlap) ** 2


def get_ideal_bitflip(wf_initial):
    """
    The ideal bit-flip is a perfect spatial mirror of the initial state.
    Assumes the spatial grid is symmetric around x=0.
    """
    return wf_initial.psi[::-1]


def get_ideal_hadamard(wf_initial):
    """
    The ideal Hadamard is a 50/50 superposition of the initial and mirrored state.
    Based on the theory, the relative phase difference is typically pi/2 (i).
    """
    mirror_psi = wf_initial.psi[::-1]
    # 1/sqrt(2) * (|initial> + i|mirrored>)
    return (wf_initial.psi - 1j * mirror_psi) / np.sqrt(2)
