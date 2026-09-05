import numpy as np
from scipy.linalg import eig, eigh


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

        H = H + np.diag(self.potential)

        if np.iscomplexobj(self.potential):
            evals, evecs = eig(H)

            idx = np.argsort(np.real(evals))
            evals = evals[idx]
            evecs = evecs[:, idx]
        else:
            evals, evecs = eigh(H)

        evecs = evecs / np.sqrt(self.dx)
        return evals, evecs

    def set_initial_state(self, wavefunction):
        """Projects the initial spatial state onto the energy eigenstates."""

        if np.iscomplexobj(self.potential):
            # For non-Hermitian systems, eigenvectors are not orthogonal.
            # We solve the linear system: evecs * c_n = psi
            self.c_n = np.linalg.solve(self.evecs, wavefunction.psi)
        else:
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


class SplitOperatorPropagator:
    """
    Evolve the wavefunction using the Split-Operator Fourier Transform (SOFT) method.
    """
    def __init__(self, grid, potential_array, m=1.0, hbar=1.0 ):
        self.grid = grid
        self.dx = grid.dx
        self.N = grid.N
        self.potential = potential_array.astype(complex)
        self.m = m
        self.hbar = hbar

        spatial_freqs = np.fft.fftfreq(self.N, d=self.dx) 
        self.p = 2.0 * np.pi * self.hbar * spatial_freqs

        self.kinetic_energy = (self.p**2) / (2.0 * self.m)
        self.initial_psi = None
        self.current_t = 0.0
        self.current_psi = None

    def set_initial_state(self, wavefunction):
        """
        Sets the initial wavefunction.
        No matrix diagonalization or projection is needed for the SOFT method.
        """
        self.initial_psi = wavefunction.psi.copy().astype(complex)
        self.current_t = 0.0
        self.current_psi = self.initial_psi.copy()

    def evolve(self, t, wavefunction_out, dt=0.005):
        """
        Evolves the state to absolute time 't'. 
        Uses caching to only compute the delta from the current time.
        """
        if self.initial_psi is None:
            raise ValueError("Initial state not set. Call set_initial_state() first.")
            
        
        if t < self.current_t or t == 0:
            self.current_t = 0.0
            self.current_psi = self.initial_psi.copy()
            
       
        time_to_evolve = t - self.current_t
        
        if time_to_evolve > 0:
            
            steps = max(1, int(time_to_evolve / dt))
            actual_dt = time_to_evolve / steps  
            
            U_V_half = np.exp(-0.5j * self.potential * actual_dt / self.hbar)
            U_K = np.exp(-1j * self.kinetic_energy * actual_dt / self.hbar)
            
            psi = self.current_psi
            for _ in range(steps):
                psi = psi * U_V_half
                psi = np.fft.ifft(np.fft.fft(psi) * U_K)
                psi = psi * U_V_half
                
           
            self.current_psi = psi
            self.current_t = t
            
       
        wavefunction_out.psi = self.current_psi.copy()

class SplitOperatorPropagator2D:
    """
    Evolves a 2D wavefunction (two particles in 1D) using the 
    Split-Operator Fourier Transform (SOFT) method.
    """
    def __init__(self, grid2d, potential_array, m=1.0, hbar=1.0):
        self.grid = grid2d
        self.dx = grid2d.dx
        self.N = grid2d.N
        self.potential = potential_array.astype(complex)
        self.m = m
        self.hbar = hbar
        
       
        spatial_frequencies = np.fft.fftfreq(self.N, d=self.dx)
        p_base = 2.0 * np.pi * self.hbar * spatial_frequencies
        
        
        self.p1, self.p2 = np.meshgrid(p_base, p_base, indexing='ij')
        
       
        self.kinetic_energy = (self.p1**2 + self.p2**2) / (2.0 * self.m)
        
        self.initial_psi = None
        self.current_t = 0.0
        self.current_psi = None

    def set_initial_state(self, wavefunction_2d):
        """Saves the exact initial state and resets the internal clock."""
        self.initial_psi = wavefunction_2d.psi.copy().astype(complex)
        self.current_t = 0.0
        self.current_psi = self.initial_psi.copy()

    def evolve(self, t, wavefunction_out, dt=0.005):
        """Evolves the 2D state forward in time."""
        if self.initial_psi is None:
            raise ValueError("Initial state not set.")
            
        if t < self.current_t or t == 0:
            self.current_t = 0.0
            self.current_psi = self.initial_psi.copy()
            
        time_to_evolve = t - self.current_t
        
        if time_to_evolve > 0:
            steps = max(1, int(time_to_evolve / dt))
            actual_dt = time_to_evolve / steps
            
            U_V_half = np.exp(-0.5j * self.potential * actual_dt / self.hbar)
            U_K = np.exp(-1j * self.kinetic_energy * actual_dt / self.hbar)
            
            psi = self.current_psi
            for _ in range(steps):
                psi = psi * U_V_half
                psi_p = np.fft.fft2(psi)
                psi_p = psi_p * U_K
                psi = np.fft.ifft2(psi_p)
                psi = psi * U_V_half
                
            self.current_psi = psi
            self.current_t = t
            
        wavefunction_out.psi = self.current_psi.copy()
    
    def update_potential(self, new_potential_array):
        """
        Dynamically updates the Hamiltonian landscape mid-simulation 
        to apply different quantum gates.
        """
        self.potential = new_potential_array.astype(complex)
