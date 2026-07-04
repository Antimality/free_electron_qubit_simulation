import numpy as np


def infinite_box(grid, L_box):
    """A flat well with infinitely high walls."""
    U = np.zeros(grid.N)
    U[np.abs(grid.x) > L_box] = 1e6  # Approximating infinity to confine the particle
    return U


def harmonic_oscillator(grid, omega):
    """A standard parabolic trap."""
    return 0.5 * (omega**2) * grid.x**2


def inharmonic_oscillator(grid, omega, gamma):
    """A standard parabolic trap with an x^4 disturbance."""
    return 0.5 * (omega**2) * grid.x**2 + gamma * grid.x**4


def ponderomotive_trap(grid, U0, sigma):
    """
    Effective potential from the focused laser field.
    U(x) ~ U0 * (1 - exp(-x^2 / sigma^2)) globally, yielding a harmonic well near x=0.
    """
    return U0 * (1 - np.exp(-(grid.x**2) / (2 * sigma**2)))


def double_well(grid, omega, mu):
    """
    A double-well potential merging two harmonic oscillators.
    Localized at x = -mu (|0>) and x = +mu (|1>).
    """
    # Using the piecewise definition from the Parienty paper
    left_well = (grid.x + mu) ** 2
    right_well = (grid.x - mu) ** 2
    return 0.5 * (omega**2) * np.minimum(left_well, right_well)


def double_gaussian(grid, U0, sigma, mu):
    """
    A double-Gaussian potential with two centered peaks or wells.
    The centers are located at x = -mu and x = +mu.
    """
    left_peak = np.exp(-((grid.x + mu) ** 2) / (2 * sigma**2))
    right_peak = np.exp(-((grid.x - mu) ** 2) / (2 * sigma**2))
    return U0 * (left_peak + right_peak)

