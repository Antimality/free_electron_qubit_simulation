import numpy as np
from scipy.linalg import eigh


class Propagator:
    """
    Evolves the wavefunction using the Semi-Analytic Eigenbasis Expansion method.
    """

    def __init__(self, grid, potential_array, m=1, hbar=1):
        self.grid = grid
        self.dx = grid.dx
        self.N = grid.N
        self.potential = potential_array
        self.m = m
        self.hbar = hbar

        # Diagonalize upon instantiation
        self.evals, self.evecs = self._diagonalize()
        self.c_n = None  # Expansion coefficients

    def _diagonalize(self):
        # Calculate the proper kinetic coefficient: hbar^2 / 2m
        coeff = (self.hbar**2) / (2.0 * self.m * self.dx**2)

        # Apply to finite difference matrix
        diag = np.ones(self.N) * (2.0 * coeff)
        off_diag = np.ones(self.N - 1) * (-1.0 * coeff)
        H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

        H += np.diag(self.potential)

        evals, evecs = eigh(H)
        evecs = evecs / np.sqrt(self.dx)
        return evals, evecs

    def set_initial_state(self, wavefunction):
        """Projects the initial spatial state onto the energy eigenstates."""
        # c_n = integral( evec_n^* * psi(x,0) ) dx
        self.c_n = np.dot(np.conj(self.evecs.T), wavefunction.psi) * self.dx

    def evolve(self, t, wavefunction_out):
        """Evolves the coefficients analytically and reconstructs the spatial state."""
        if self.c_n is None:
            raise ValueError("Initial state not set. Call set_initial_state() first.")

        # Exact phase evolution: c_n(t) = c_0 * exp(-i * E_n * t / hbar)
        phases = np.exp(-1j * self.evals * t / self.hbar)
        c_nt = self.c_n * phases

        # Reconstruct psi(x,t) = sum(c_n(t) * evec_n(x))
        wavefunction_out.psi = np.dot(self.evecs, c_nt)
