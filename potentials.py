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


def double_gaussian(grid, U0, sigma):
    """
    A double-Gaussian potential with two centered peaks or wells.
    The centers are located at x = -L/2 and x = L/2
    """
    left_peak = np.exp(-((grid.x + grid.L / 2) ** 2) / (2 * sigma**2))
    right_peak = np.exp(-((grid.x - grid.L / 2) ** 2) / (2 * sigma**2))
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
