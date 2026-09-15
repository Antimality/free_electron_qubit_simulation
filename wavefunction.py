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

    def init_coherent_state(self, offset, omega, m=1.0, hbar=1.0, alpha_phase=0.0):
        """
        Initializes the wavefunction as a minimum-uncertainty coherent state.

        'mu' represents the displacement offset from the CENTER of the grid
        (e.g., mu > 0 places it in the right well, mu < 0 in the left well).
        """
        x = self.grid.x

        # 1. Dynamically find the center of the new [0, L] grid
        # (Using x[-1] safely gets the max boundary regardless of how the Grid class is structured)
        grid_center = x[-1] / 2.0

        # 2. Shift the requested offset to absolute grid coordinates
        absolute_mu = grid_center + offset

        # Calculate the characteristic oscillator width scale (alpha parameter)
        alpha_width = np.sqrt(m * omega / hbar)

        # 3. Use the absolute_mu for the exponent
        exponent = -0.5 * (alpha_width * (x - absolute_mu)) ** 2

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

    def init_sine_wave(self, mu, delta, n_mode=1):
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
        xmax = mu + delta
        xmin = mu - delta
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

    def init_random_packet(self, mu, delta, max_modes=15):
        """
        Initializes the wavefunction as a physically valid random wave packet
        confined strictly around 'mu' with a spacial variation 'sigma'.

        It constructs a superposition of localized box modes with randomized
        amplitudes and phases, ensuring it behaves elegantly under time evolution.
        """
        x = self.grid.x
        self.psi = np.zeros_like(x, dtype=complex)

        xmax = mu + delta
        xmin = mu - delta

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
        """Probability amplitude of the |0> state (left half of the grid)."""
        grid_center = self.grid.x[-1] / 2.0
        mask = self.grid.x < grid_center
        return np.sum(self.get_probability_density()[mask]) * self.grid.dx

    def get_right_probability(self):
        """Probability amplitude of the |1> state (right half of the grid)."""
        grid_center = self.grid.x[-1] / 2.0
        mask = self.grid.x >= grid_center
        return np.sum(self.get_probability_density()[mask]) * self.grid.dx

    def get_qutip_qubit(self):
        """
        Converts the continuous spatial wavefunction into a 2-level QuTiP Qobj (ket),
        ready to be visualized on a Bloch sphere.
        """
        try:
            import qutip as qt
        except ImportError:
            raise ImportError("The 'qutip' library is required.")

        mask0 = self.grid.x < self.grid.center
        mask1 = self.grid.x >= self.grid.center

        density = self.get_probability_density()

        p0 = np.sum(density[mask0]) * self.grid.dx
        p1 = np.sum(density[mask1]) * self.grid.dx

        # Average phase for each qubit
        phase_0 = np.angle(np.sum(self.psi[mask0])) if p0 > 1e-12 else 0.0
        phase_1 = np.angle(np.sum(self.psi[mask1])) if p1 > 1e-12 else 0.0

        # Construct the amplitudes while preserving phase information (crucial for equatorial motion)
        alpha = np.sqrt(p0) * np.exp(1j * phase_0)
        beta = np.sqrt(p1) * np.exp(1j * phase_1)

        qubit_state = qt.Qobj([[alpha], [beta]])

        return qubit_state.unit()


class Wavefunction2D:
    """
    Represents the quantum state of two particles in a 2D configuration space.
    """

    def __init__(self, grid2d):
        self.grid = grid2d
        self.psi = np.zeros((grid2d.N, grid2d.N), dtype=complex)

    def _normalize(self):
        """Helper to strictly normalize the 2D wavefunction."""
        dx = self.grid.dx
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * (dx**2))
        if norm > 1e-15:
            self.psi = (self.psi / norm).astype(complex)

    def init_gaussian(self, mu1, mu2, sigma):
        """
        Initializes a normalized 2D Gaussian wavepacket (Product state).
        mu1: position of particle 1
        mu2: position of particle 2
        """

        g1 = np.exp(-((self.grid.x1 - mu1) ** 2) / (4 * sigma**2))
        g2 = np.exp(-((self.grid.x2 - mu2) ** 2) / (4 * sigma**2))
        self.psi = (g1 * g2).astype(complex)
        self._normalize()

    def init_fermionic_gaussian(self, mu1, mu2, sigma):
        """
        BONUS: Initializes a perfectly antisymmetric state (Slater determinant)
        required if the two particles are identical electrons with the same spin.
        """
        g1_A = np.exp(-((self.grid.x1 - mu1) ** 2) / (4 * sigma**2))
        g2_A = np.exp(-((self.grid.x2 - mu2) ** 2) / (4 * sigma**2))

        g1_B = np.exp(-((self.grid.x1 - mu2) ** 2) / (4 * sigma**2))
        g2_B = np.exp(-((self.grid.x2 - mu1) ** 2) / (4 * sigma**2))

        self.psi = (g1_A * g2_A - g1_B * g2_B).astype(complex)
        self._normalize()

    def init_coherent_state(
        self, offset1, offset2, omega, m=1.0, hbar=1.0, alpha_phase=0.0
    ):
        """
        Initializes a 2D minimum-uncertainty coherent state.
        """

        grid_center = (self.grid.x[0] + self.grid.x[-1]) / 2.0

        absolute_mu1 = grid_center + offset1
        absolute_mu2 = grid_center + offset2

        alpha_width = np.sqrt(m * omega / hbar)

        exponent = (
            -0.5 * (alpha_width * (self.grid.x1 - absolute_mu1)) ** 2
            - 0.5 * (alpha_width * (self.grid.x2 - absolute_mu2)) ** 2
        )

        self.psi = (np.exp(exponent) * np.exp(1j * alpha_phase)).astype(complex)
        self._normalize()

    def init_sine_wave(self, mu1, mu2, delta, n_mode1=1, n_mode2=1):
        """
        Initializes the wavefunction as a 2D localized sine wave mode
        (particle in a 2D box).
        """
        self.psi = np.zeros_like(self.grid.x1, dtype=complex)

        xmin1, xmax1 = mu1 - delta, mu1 + delta
        xmin2, xmax2 = mu2 - delta, mu2 + delta

        L_local = 2 * delta
        if L_local <= 0:
            raise ValueError("delta must be positive to define a local box.")

        inside_mask = (
            (self.grid.x1 >= xmin1)
            & (self.grid.x1 <= xmax1)
            & (self.grid.x2 >= xmin2)
            & (self.grid.x2 <= xmax2)
        )

        k1 = n_mode1 * np.pi / L_local
        k2 = n_mode2 * np.pi / L_local

        amplitude = 2.0 / L_local  # sqrt(2/L) * sqrt(2/L)

        self.psi[inside_mask] = (
            amplitude
            * np.sin(k1 * (self.grid.x1[inside_mask] - xmin1))
            * np.sin(k2 * (self.grid.x2[inside_mask] - xmin2))
        )
        self._normalize()

    def init_random_packet(self, mu1, mu2, delta, max_modes=5):
        """
        Initializes a physically valid random 2D wave packet
        confined strictly around (mu1, mu2).
        """
        self.psi = np.zeros_like(self.grid.x1, dtype=complex)

        xmin1, xmax1 = mu1 - delta, mu1 + delta
        xmin2, xmax2 = mu2 - delta, mu2 + delta
        L_local = 2 * delta

        inside_mask = (
            (self.grid.x1 >= xmin1)
            & (self.grid.x1 <= xmax1)
            & (self.grid.x2 >= xmin2)
            & (self.grid.x2 <= xmax2)
        )

        num_local_points = np.sum(inside_mask)
        if num_local_points == 0:
            raise ValueError("No grid points found within the requested interval.")

        for n1 in range(1, max_modes + 1):
            for n2 in range(1, max_modes + 1):
                c_n = np.random.normal() + 1j * np.random.normal()
                k1 = n1 * np.pi / L_local
                k2 = n2 * np.pi / L_local

                mode_profile = (
                    (2.0 / L_local)
                    * np.sin(k1 * (self.grid.x1[inside_mask] - xmin1))
                    * np.sin(k2 * (self.grid.x2[inside_mask] - xmin2))
                )
                self.psi[inside_mask] += c_n * mode_profile

        self._normalize()

    def get_probability_density(self):
        """Returns |psi(x1, x2)|^2."""
        return np.abs(self.psi) ** 2

    def _get_quadrant_masks(self):
        """Helper to divide the 2D grid into 4 logical states."""
        grid_center = (self.grid.x[0] + self.grid.x[-1]) / 2.0

        # x1 < center => Particle 1 in |0> (left)
        # x1 >= center => Particle 1 in |1> (right)
        mask_00 = (self.grid.x1 < grid_center) & (self.grid.x2 < grid_center)
        mask_01 = (self.grid.x1 < grid_center) & (self.grid.x2 >= grid_center)
        mask_10 = (self.grid.x1 >= grid_center) & (self.grid.x2 < grid_center)
        mask_11 = (self.grid.x1 >= grid_center) & (self.grid.x2 >= grid_center)

        return mask_00, mask_01, mask_10, mask_11

    def get_logical_probabilities(self):
        """Returns the probability distribution [P_00, P_01, P_10, P_11]."""
        density = self.get_probability_density()
        dx2 = self.grid.dx**2
        masks = self._get_quadrant_masks()

        return [np.sum(density[m]) * dx2 for m in masks]

    def get_qutip_qubit(self):
        """
        Converts the 2D continuous spatial wavefunction into a 2-qubit
        QuTiP Qobj (ket) tensor product state, ready for 2-qubit operations.
        """
        try:
            import qutip as qt
        except ImportError:
            raise ImportError("The 'qutip' library is required.")

        probs = self.get_logical_probabilities()
        masks = self._get_quadrant_masks()

        amplitudes = []
        for p, mask in zip(probs, masks):
            phase = np.angle(np.sum(self.psi[mask])) if p > 1e-12 else 0.0
            amplitudes.append(np.sqrt(p) * np.exp(1j * phase))

        qubit_state = qt.Qobj(np.array(amplitudes).reshape(4, 1), dims=[[2, 2], [1, 1]])

        return qubit_state.unit()
