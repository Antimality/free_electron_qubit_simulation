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


def coulomb_and_infinite_box(grid, interacting_mu, k_coulomb=0.1, softening=0.05):
    """
    An infinite box potential combined with a 1D soft-core Coulomb interaction.
    
    Parameters:
    - interacting_mu: The position of the *other* qubit (the source of the Coulomb force).
    - k_coulomb: Interaction strength (q1*q2). Positive values mean repulsion.
    - softening: A small parameter to prevent division by zero at x = interacting_mu.
    """
    # 1. התחלה עם פוטנציאל הבור האינסופי
    U_total = infinite_box(grid).copy()
    
    # 2. הוספת פוטנציאל קולון המרוכך
    U_coulomb = k_coulomb / np.sqrt((grid.x - interacting_mu)**2 + softening**2)
    
    # חיבור הפוטנציאלים. מחוץ לבור הפוטנציאל יישאר 1e6 (בגלל הבור האינסופי)
    U_total += U_coulomb
    
    return U_total

def asymmetric_coulomb_double_well(grid, L_box, interacting_mu, depth1=-500.0, depth2=-300.0, mu1=None, mu2=None, well_sigma=0.05, k_coulomb=0.1, softening=0.05):
    """
    An asymmetric double-Gaussian well combined with Coulomb interaction.
    depth1 and depth2 control the depth of each well independently to break symmetry.
    """
    if mu1 is None:
        mu1 = L_box * 0.25
    if mu2 is None:
        mu2 = L_box * 0.75

    # 1. התחלה עם פוטנציאל אפס וגבולות אינסופיים
    U_total = np.zeros(grid.N)
    out_of_bounds = (grid.x < grid.x[0]) | (grid.x > grid.x[-1])
    
    # 2. יצירת שני הבורות הגאוסיאניים (ערכים שליליים כדי ליצור בור)
    well1 = depth1 * np.exp(-((grid.x - mu1)**2) / (2 * well_sigma**2))
    well2 = depth2 * np.exp(-((grid.x - mu2)**2) / (2 * well_sigma**2))
    
    # חיבור הבורות לפוטנציאל
    U_total[~out_of_bounds] += well1[~out_of_bounds] + well2[~out_of_bounds]
    
    # חסימת הקצוות לאינסוף
    U_total[out_of_bounds] = 1e6
    
    # 3. הוספת פוטנציאל קולון מרוכך מהקיוביט השני
    U_coulomb = k_coulomb / np.sqrt((grid.x - interacting_mu)**2 + softening**2)
    U_total += U_coulomb
    
    return U_total
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
