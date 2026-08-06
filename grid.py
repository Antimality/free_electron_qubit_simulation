import numpy as np


class Grid:
    """
    Defines the 1D spatial grid for the simulation.
    """

    def __init__(self, L, N, padding_mult):
        self.L = L  # length of the grid (domain is 0 to L)
        self.N = N  # Number of discrete spatial points
        self.x = np.linspace(-L * padding_mult, L * padding_mult, N)
        self.dx = self.x[1] - self.x[0]
