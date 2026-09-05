import numpy as np


class Grid:
    """
    Defines the 1D spatial grid for the simulation.
    """

    def __init__(self, L, N, padding_mult=1.0):
        self.L = L  # length of the grid (domain is -L/2 to L/2)
        self.N = N  # Number of discrete spatial points

        total_span = L * padding_mult
        self.x = np.linspace(-total_span / 2, total_span / 2, N)
        self.dx = self.x[1] - self.x[0]

        self.x_max, self.x_min = self.x[-1], self.x[0]
        self.grid_center = (self.x_max + self.x_min) / 2
