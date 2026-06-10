import numpy as np


class Wavefunction:
    """
    Represents the quantum state of the electron.
    """

    def __init__(self, grid):
        self.grid = grid
        self.psi = np.zeros(grid.N, dtype=complex)

    def init_gaussian(self, mu, sigma):
        """Initializes a normalized Gaussian wavepacket shifted by mu."""
        psi = np.exp(-((self.grid.x - mu) ** 2) / (4 * sigma**2))
        # Normalize the wavefunction: integral(|psi|^2)dx = 1
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * self.grid.dx)
        self.psi = (psi / norm).astype(complex)

    def init_coherent_state(self, mu, omega, m=1.0, hbar=1.0, alpha_phase=0.0):
        """
        Initializes the wavefunction as a minimum-uncertainty coherent state
        (a displaced ground-state Gaussian) centered at position 'mu'.

        As detailed in the theoretical framework, under a harmonic potential
        of frequency 'omega', this spatial wave packet maintains its shape
        without spreading during idle hold periods.
        """
        x = self.grid.x

        # Calculate the characteristic oscillator width scale (alpha parameter)
        # alpha_width = sqrt(m * omega / hbar)
        alpha_width = np.sqrt(m * omega / hbar)

        # Displacement magnitude matches the center 'mu' in units of oscillator length
        # A displaced ground state profile: psi(x) ~ exp(-0.5 * (alpha_width * (x - mu))^2)
        exponent = -0.5 * (alpha_width * (x - mu)) ** 2

        # Incorporate optional phase rotation arg(alpha)
        self.psi = (np.exp(exponent) * np.exp(1j * alpha_phase)).astype(complex)

        # Strictly normalize the physical state across the grid
        dx = (
            self.grid.x[1] - self.grid.x[0]
            if hasattr(self.grid, "dx")
            else (x[1] - x[0])
        )
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * dx)
        self.psi /= norm

    def init_sine_wave(self, xmin, xmax, n_mode=1):
        """
        Initializes the wavefunction as a localized sine wave mode
        corresponding to an infinite square well (particle in a box) eigenstate
        bounded tightly between 'xmin' and 'xmax'.

        phi_n(x) = sqrt(2 / L_local) * sin(n * pi * (x - xmin) / L_local) inside the well,
        and 0 everywhere else.
        """
        x = self.grid.x
        self.psi = np.zeros_like(x, dtype=complex)

        # Calculate localized box dimensions
        L_local = xmax - xmin
        if L_local <= 0:
            raise ValueError(
                "xmax must be strictly greater than xmin to define a local box."
            )

        # Create a mask to isolate the localized sub-plane region
        inside_mask = (x >= xmin) & (x <= xmax)

        # Calculate the sinusoidal distribution inside the boundary bounds
        k = n_mode * np.pi / L_local
        self.psi[inside_mask] = np.sqrt(2.0 / L_local) * np.sin(
            k * (x[inside_mask] - xmin)
        )

        # Double check normalization consistency across the discrete grid spacing
        dx = (
            self.grid.x[1] - self.grid.x[0]
            if hasattr(self.grid, "dx")
            else (x[1] - x[0])
        )
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * dx)
        self.psi /= norm

    def init_random_packet(self, xmin, xmax, max_modes=15):
        """
        Initializes the wavefunction as a physically valid random wave packet
        confined strictly between 'xmin' and 'xmax'.

        It constructs a superposition of localized box modes with randomized
        amplitudes and phases, ensuring it behaves elegantly under time evolution.
        """
        x = self.grid.x
        self.psi = np.zeros_like(x, dtype=complex)

        L_local = xmax - xmin
        if L_local <= 0:
            raise ValueError(
                "xmax must be strictly greater than xmin to define a localized domain."
            )

        inside_mask = (x >= xmin) & (x <= xmax)
        num_local_points = np.sum(inside_mask)

        if num_local_points == 0:
            raise ValueError(
                "No grid points found within the requested xmin and xmax interval."
            )

        # Generate random complex expansion coefficients for the chosen number of modes
        # Random amplitudes from a standard normal distribution + random phase arguments
        amplitudes = np.random.normal(size=max_modes) + 1j * np.random.normal(
            size=max_modes
        )

        # Build the wavepacket as a localized Fourier-sine series expansion
        for n_idx, c_n in enumerate(amplitudes):
            n_mode = n_idx + 1
            k = n_mode * np.pi / L_local

            # Compute the n-th box eigenstate shape
            mode_profile = np.sqrt(2.0 / L_local) * np.sin(k * (x[inside_mask] - xmin))

            # Superpose into our localized sub-plane vector slice
            self.psi[inside_mask] += c_n * mode_profile

        # --- Strict Discrete Normalization across the grid spacing ---
        # Handle custom grid properties smoothly (checking for discrete dx)
        dx = self.grid.dx if hasattr(self.grid, "dx") else (x[1] - x[0])
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * dx)

        if norm > 1e-15:
            self.psi /= norm
        else:
            # Fallback safety default to avoid dividing by absolute zero
            self.psi[inside_mask] = 1.0 / np.sqrt(num_local_points * dx)

    def get_probability_density(self):
        """Returns |psi(x)|^2."""
        return np.abs(self.psi) ** 2

    def get_left_probability(self):
        """Probability amplitude of the |0> state (x < 0)."""
        mask = self.grid.x < 0
        return np.sum(self.get_probability_density()[mask]) * self.grid.dx

    def get_right_probability(self):
        """Probability amplitude of the |1> state (x >= 0)."""
        mask = self.grid.x >= 0
        return np.sum(self.get_probability_density()[mask]) * self.grid.dx
