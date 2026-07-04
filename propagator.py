import numpy as np
from scipy.linalg import eigh


class Propagator:
    """
    Evolves the wavefunction using the Semi-Analytic Eigenbasis Expansion method.
    """

    def __init__(self, grid, potential_array, hbar=1):
        self.grid = grid
        self.dx = grid.dx
        self.N = grid.N
        self.potential = potential_array
        self.hbar = hbar

        # Diagonalize upon instantiation
        self.evals, self.evecs = self._diagonalize()
        self.c_n = None  # Expansion coefficients

    def _diagonalize(self):
        """Constructs the Hamiltonian and finds its eigenvalues and eigenvectors."""
        # 1. Kinetic Energy Matrix (Tridiagonal finite difference for d^2/dx^2)
        # K = - (1/2m) * d^2/dx^2  (assuming m=1)
        diag = np.ones(self.N) * (1.0 / self.dx**2)
        off_diag = np.ones(self.N - 1) * (-0.5 / self.dx**2)
        H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

        # 2. Add Potential Energy
        H += np.diag(self.potential)

        # 3. Solve for Eigenvalues (E_n) and Eigenvectors (psi_n)
        evals, evecs = eigh(H)

        # scipy eigh normalizes Euclidean vectors (sum V^2 = 1).
        # We need continuous space normalization (int |V|^2 dx = 1)
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
