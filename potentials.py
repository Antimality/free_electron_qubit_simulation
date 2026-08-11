import numpy as np



def infinite_box(grid):
    """A flat well with infinitely high walls from 0 to L_grid."""
    U = np.zeros(grid.N)

    # Set potential to 1e6 (infinity) anywhere outside the [0, L_grid] range
    U[(grid.x < grid.x[0]) | (grid.x > grid.x[-1])] = 1e6

    return U


def harmonic_oscillator(grid, m, omega):
    """A standard parabolic trap."""
    return 0.5 * m * (omega**2) * (grid.x) **2


def inharmonic_oscillator(grid, m, omega, gamma):
    """A standard parabolic trap with an x^4 disturbance."""
    return harmonic_oscillator(grid, m, omega) + gamma * (grid.x - grid.L / 2) ** 4


def double_gaussian(grid, U0, sigma, mu):
    """
    A double-Gaussian potential with two centered peaks or wells.
    The centers are located symmetrically at x = -mu and x = +mu.
    """
    left_peak = np.exp(-((grid.x + mu) ** 2) / (2 * sigma**2))
    right_peak = np.exp(-((grid.x - mu) ** 2) / (2 * sigma**2))
    return U0 * (left_peak + right_peak)


def complex_absorbing_potential(grid, width=0.2, strength=100.0):
    """
    A complex absorbing potential (CAP) to simulate open boundary conditions.
    The CAP is applied at the edges of the grid to absorb outgoing waves.
    """
    U = np.zeros(grid.N, dtype=complex)

    left_bound = grid.x[0] + width
    left_mask = grid.x < left_bound
    U[left_mask] = strength * ((left_bound - grid.x[left_mask]) / width) ** 3

    right_bound = grid.x[-1] - width
    right_mask = grid.x > right_bound
    U[right_mask] = strength * ((grid.x[right_mask] - right_bound) / width) ** 3

    return U

def complex_absorbing_potential_2d(grid2d, width=0.2, strength=100.0):
    """
    Creates a 2D Complex Absorbing Potential (CAP) at the grid boundaries
    to absorb the wavefunction if EITHER particle reaches the edges.
    """
    W = np.zeros((grid2d.N, grid2d.N))
    
    left_bound = grid2d.x[0] + width
    right_bound = grid2d.x[-1] - width
    
    # --- Absorption for Particle 1 (x1 axis) ---
    left_mask_1 = grid2d.x1 < left_bound
    W[left_mask_1] += strength * ((left_bound - grid2d.x1[left_mask_1]) / width)**3
    
    right_mask_1 = grid2d.x1 > right_bound
    W[right_mask_1] += strength * ((grid2d.x1[right_mask_1] - right_bound) / width)**3

    # --- Absorption for Particle 2 (x2 axis) ---
    left_mask_2 = grid2d.x2 < left_bound
    W[left_mask_2] += strength * ((left_bound - grid2d.x2[left_mask_2]) / width)**3
    
    right_mask_2 = grid2d.x2 > right_bound
    W[right_mask_2] += strength * ((grid2d.x2[right_mask_2] - right_bound) / width)**3
    
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
    distance_squared = (grid2d.x1 - grid2d.x2)**2
    return k_e / np.sqrt(distance_squared + softening_a**2)



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
