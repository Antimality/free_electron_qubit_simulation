import numpy as np


def infinite_box(grid):
    """A flat well with infinitely high walls from -L/2 to L/2."""
    U = np.zeros(grid.N)

    # Set potential to 1e6 (infinity) anywhere outside the [-L/2, L/2] range
    U[(grid.x < -grid.L / 2) | (grid.x > grid.L / 2)] = 1e6

    return U


def harmonic_oscillator(grid, m, omega):
    """A standard parabolic trap."""
    return 0.5 * m * (omega**2) * grid.x**2


def inharmonic_oscillator(grid, m, omega, gamma):
    """A standard parabolic trap with an x^4 disturbance."""
    return harmonic_oscillator(grid, m, omega) + gamma * grid.x**4


def double_gaussian(grid, U0, sigma, mu):
    """
    A double-Gaussian potential with two centered peaks or wells.
    The centers are located symmetrically at x = -mu and x = +mu.
    """
    left_peak = np.exp(-((grid.x + mu) ** 2) / (2 * sigma**2))
    right_peak = np.exp(-((grid.x - mu) ** 2) / (2 * sigma**2))
    return U0 * (left_peak + right_peak)


def gaussian_walls(grid, U0, sigma, wall_pos):
    """
    Creates an approximate infinite well using two Gaussian peaks.
    """
    # Left wall at -wall_pos
    left_wall = np.exp(-((grid.x + wall_pos) ** 2) / (2 * sigma**2))

    # Right wall at +wall_pos
    right_wall = np.exp(-((grid.x - wall_pos) ** 2) / (2 * sigma**2))

    return U0 * (left_wall + right_wall)


def complex_absorbing_potential(grid, width=0.2, strength=100.0):
    """
    A complex absorbing potential (CAP) to simulate open boundary conditions.
    The CAP is applied at the edges of the grid to absorb outgoing waves.
    """
    U = np.zeros(grid.N, dtype=complex)

    left_bound = grid.x_min + width
    left_mask = grid.x < left_bound
    U[left_mask] = strength * ((left_bound - grid.x[left_mask]) / width) ** 3

    right_bound = grid.x_max - width
    right_mask = grid.x > right_bound
    U[right_mask] = strength * ((grid.x[right_mask] - right_bound) / width) ** 3

    return U


def complex_absorbing_potential_2d(grid2d, width=0.2, strength=100.0):
    """
    Creates a 2D Complex Absorbing Potential (CAP) at the grid boundaries
    to absorb the wavefunction if EITHER particle reaches the edges.
    """
    W = np.zeros((grid2d.N, grid2d.N))

    left_bound = grid2d.x_min + width
    right_bound = grid2d.x_max - width

    # --- Absorption for Particle 1 (x1 axis) ---
    left_mask_1 = grid2d.x1 < left_bound
    W[left_mask_1] += strength * ((left_bound - grid2d.x1[left_mask_1]) / width) ** 3

    right_mask_1 = grid2d.x1 > right_bound
    W[right_mask_1] += strength * ((grid2d.x1[right_mask_1] - right_bound) / width) ** 3

    # --- Absorption for Particle 2 (x2 axis) ---
    left_mask_2 = grid2d.x2 < left_bound
    W[left_mask_2] += strength * ((left_bound - grid2d.x2[left_mask_2]) / width) ** 3

    right_mask_2 = grid2d.x2 > right_bound
    W[right_mask_2] += strength * ((grid2d.x2[right_mask_2] - right_bound) / width) ** 3

    return W


def double_well_2d(grid2d, U0, sigma, mu):
    """
    Applies the double well potential independently to both particles.
    V_total = V(x1) + V(x2)
    """

    left_1 = np.exp(-((grid2d.x1 + mu) ** 2) / (2 * sigma**2))
    right_1 = np.exp(-((grid2d.x1 - mu) ** 2) / (2 * sigma**2))
    V1 = U0 * (left_1 + right_1)

    left_2 = np.exp(-((grid2d.x2 + mu) ** 2) / (2 * sigma**2))
    right_2 = np.exp(-((grid2d.x2 - mu) ** 2) / (2 * sigma**2))
    V2 = U0 * (left_2 + right_2)

    return V1 + V2


def coulomb_interaction(grid2d, k_e=1.0, softening_a=0.1):
    """
    Soft-core Coulomb repulsion between the two particles.
    V = k_e / sqrt((x1 - x2)^2 + a^2)
    """
    distance_squared = (grid2d.x1 - grid2d.x2) ** 2
    return k_e / np.sqrt(distance_squared + softening_a**2)


def adjacent_double_wells_2d(grid2d, U0_1, U0_2, sigma, mu, D, dU1 = 0, dU2 = 0):
    """
    Creates two separate double-wells in 1D physical space.
    Electron 1 is confined to the left double-well (centered at -D/2).
    Electron 2 is confined to the right double-well (centered at +D/2).
    """
    # Qubit 1 binding well
    left_1 = np.exp(-((grid2d.x1 - (-D / 2 - mu)) ** 2) / (2 * sigma**2))
    right_1 = np.exp(-((grid2d.x1 - (-D / 2 + mu)) ** 2) / (2 * sigma**2))
    V1 = U0_1 * left_1 + (U0_1 + dU1) * right_1

    # Qubit 2 binding well
    left_2 = np.exp(-((grid2d.x2 - (D / 2 - mu)) ** 2) / (2 * sigma**2))
    right_2 = np.exp(-((grid2d.x2 - (D / 2 + mu)) ** 2) / (2 * sigma**2))
    V2 = U0_2 * left_2 + (U0_2 + dU2) * right_2

    return V1 + V2
