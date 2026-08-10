import time
import numpy as np

# Import custom modules
from grid import Grid

from potentials import coulomb_and_infinite_box
from wavefunction import Wavefunction
from propagator import Propagator
from visualizations_2 import (
    animate_combined_simulation_2q,
)
from gate_finder import find_bit_flip_time, find_hadamard_time


if __name__ == "__main__":
    # 1. Constants
    L_grid = 1.0
    N_grid = 1000
    m = 1.0
    hbar = 1.0
    
    # מיקומים התחלתיים לשני הקיוביטים
    mu_1 = L_grid * 1 / 4  # קיוביט ראשון בצד שמאל
    mu_2 = L_grid * 3 / 4  # קיוביט שני בצד ימין
    sigma = 0.05

    # 2. Setup Grid
    grid = Grid(L_grid, N_grid)

    # 3. Initialize states
    np.random.seed(42)
    
    wf1_initial = Wavefunction(grid)
    wf1_initial.init_gaussian(mu=mu_1, sigma=sigma)
    
    wf2_initial = Wavefunction(grid)
    wf2_initial.init_gaussian(mu=mu_2, sigma=sigma)

    # 4. Potentials
    
    print("Calculating potentials including Coulomb interactions...")
    potential_1 = coulomb_and_infinite_box(grid, L_box=L_grid, interacting_mu=mu_2, k_coulomb=0.1)
    potential_2 = coulomb_and_infinite_box(grid, L_box=L_grid, interacting_mu=mu_1, k_coulomb=0.1)

    # 5. Initialize the Propagator engines
    print("Diagonalizing Hamiltonians (this may take a few seconds)...")
    start_time = time.time()
    
    prop1 = Propagator(grid, potential_1)
    prop2 = Propagator(grid, potential_2)

    # Set the initial states for the propagators!
    prop1.set_initial_state(wf1_initial)
    prop2.set_initial_state(wf2_initial)
    
    print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

    # 6. Calculate theoretical gate times
    E1 = prop1.evals[0]
    T_h = (np.pi * hbar) / (2 * E1)
    T_f = 2 * T_h

    # 7. Evolve 
    wf1_h = Wavefunction(grid)
    wf1_f = Wavefunction(grid)
    
    wf2_h = Wavefunction(grid)
    wf2_f = Wavefunction(grid)

    max_dens1 = np.max(np.abs(wf1_initial.psi)**2)
    max_dens2 = np.max(np.abs(wf2_initial.psi)**2)

    optimal_y_lim = max(max_dens1, max_dens2) * 1.2



    print("\nStarting combined split-screen animation for 2 Qubits...")
    

    animate_combined_simulation_2q(
        grid,
        prop1,
        prop2,
        wf1_initial,
        wf2_initial,
        total_time=T_f * 2,
        fps=30,
        y_lim=optimal_y_lim,
    )