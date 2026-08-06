import numpy as np


def infinite_box(grid):
    """A flat well with infinitely high walls from 0 to L_grid."""
    U = np.zeros(grid.N)

    # Set potential to 1e6 (infinity) anywhere outside the [0, L_grid] range
    U[(grid.x < 0) | (grid.x > grid.L)] = 1e6

    return U


def harmonic_oscillator(grid, m, omega):
    """A standard parabolic trap."""
    return 0.5 * m * (omega**2) * (grid.x - grid.L / 2) ** 2


def inharmonic_oscillator(grid, m, omega, gamma):
    """A standard parabolic trap with an x^4 disturbance."""
    return harmonic_oscillator(grid, m, omega) + gamma * (grid.x - grid.L / 2) ** 4


def double_gaussian(grid, U0, sigma):
    """
    A double-Gaussian potential with two centered peaks or wells.
    The centers are located at x = 0 and x = L
    """
    left_peak = np.exp(-((grid.x) ** 2) / (2 * sigma**2))
    right_peak = np.exp(-((grid.x - grid.L) ** 2) / (2 * sigma**2))
    return U0 * (left_peak + right_peak)


# def ponderomotive_trap(grid, U0, sigma):
#     """
#     Effective potential from the focused laser field.
#     U(x) ~ U0 * (1 - exp(-x^2 / sigma^2)) globally, yielding a harmonic well near x=0.
#     """
#     return U0 * (1 - np.exp(-(grid.x**2) / (2 * sigma**2)))


# def double_well(grid, omega, mu):
#     """
#     A double-well potential merging two harmonic oscillators.
#     Localized at x = -mu (|0>) and x = +mu (|1>).
#     """
#     # Using the piecewise definition from the Parienty paper
#     left_well = (grid.x + mu) ** 2
#     right_well = (grid.x - mu) ** 2
#     return 0.5 * (omega**2) * np.minimum(left_well, right_well)
