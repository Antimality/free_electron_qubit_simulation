import numpy as np


class Grid:
    """
    Defines the 1D spatial grid for the simulation.
    """

    def __init__(self, L, N, padding_mult=1.0):
        self.L = L  # length of the grid (domain is 0 to L)
        self.N = N  # Number of discrete spatial points

        total_span = L *padding_mult
        self.x = np.linspace(-total_span/2, total_span/2, N)
        self.dx = self.x[1] - self.x[0]

class Grid2D:
    """
    Defines a 2D spatial grid (configuration space) for the simulation of two particles.
    x1 represents the spatial position of the first particle.
    x2 represents the spatial position of the second particle.
    """
    def __init__(self, L, N, padding_mult=1.0):
        self.L = L
        self.N = N

        total_span = L * padding_mult
        self.x = np.linspace(-total_span / 2, total_span / 2, N)
        self.dx = self.x[1] - self.x[0]

        self.x1, self.x2 = np.meshgrid(self.x, self.x, indexing='ij')