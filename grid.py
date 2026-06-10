import numpy as np


class Grid:
    """
    Defines the 1D spatial grid for the simulation.
    """

    def __init__(self, L, N):
        self.L = L  # Half-length of the grid (domain is -L to L)
        self.N = N  # Number of discrete spatial points
        self.x = np.linspace(-L, L, N)
        self.dx = self.x[1] - self.x[0]
