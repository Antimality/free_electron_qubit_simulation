import time
import numpy as np

# Import custom modules
from grid import Grid

from potentials import asymmetric_coulomb_double_well
from wavefunction import Wavefunction
from propagator import Propagator
from visualizations_2 import (
    animate_combined_simulation_2q,
)
from gate_finder import find_q1_hadamard_q2_idle


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
    potential_1 = asymmetric_coulomb_double_well(grid, L_box=L_grid, interacting_mu=mu_2, k_coulomb=0.1, depth1 = -15, depth2 = -20)
    potential_2 = asymmetric_coulomb_double_well(grid, L_box=L_grid, interacting_mu=mu_1, k_coulomb=0.1, depth1 = -15, depth2 = -20)

    # 5. Initialize the Propagator engines
    print("Diagonalizing Hamiltonians (this may take a few seconds)...")
    start_time = time.time()
    
    prop1 = Propagator(grid, potential_1)
    prop2 = Propagator(grid, potential_2)

    # Set the initial states for the propagators!
    prop1.set_initial_state(wf1_initial)
    prop2.set_initial_state(wf2_initial)
    
    print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

    # 6. Search for the Joint Gate Time
    print("Searching for joint gate time: Q1 -> Hadamard, Q2 -> Idle...")
    
    # חישוב זמני מחזור תיאורטיים כדי להגדיר חלון סריקה הגיוני
    delta_E1 = prop1.evals[1] - prop1.evals[0]
    delta_E2 = prop2.evals[1] - prop2.evals[0]
    T_f1_approx = (np.pi * hbar) / delta_E1
    T_f2_approx = (np.pi * hbar) / delta_E2
    
    # נסרוק חלון זמן רחב שיכסה מספר מחזורים
    max_search_time = max(T_f1_approx, T_f2_approx) * 10000
    
    best_t, f1_hadamard, f2_idle = find_q1_hadamard_q2_idle(
        prop1, prop2, wf1_initial, wf2_initial, max_search_time
    )
    
    print(f"\n--- Optimal Joint Time Found: t = {best_t:.4f} ---")
    print(f"Qubit 1 Hadamard Fidelity: {f1_hadamard * 100:.2f}%")
    print(f"Qubit 2 Idle Fidelity:     {f2_idle * 100:.2f}%\n")
    
    # נגדיר את הזמן הכולל של האנימציה להיות קצת אחרי הזמן שמצאנו
    total_anim_time = best_t * 1.2

    # 7. Evolve 

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
        total_time=total_anim_time, 
        fps=30,
        y_lim=optimal_y_lim,
        target_t=best_t,
    )