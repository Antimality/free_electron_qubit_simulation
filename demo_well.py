import time
import numpy as np

# Import custom modules
from grid import Grid
from potentials import infinite_box
from wavefunction import Wavefunction
from propagator import Propagator
from visualizations import (
    visual_results,
    animate_simulation,
    plot_waveform_evolution_2d,
)

if __name__ == "__main__":
    # Constants
    L_grid = 1.0
    N_grid = 1000
    m = 1.0
    hbar = 1.0
    mu = L_grid * 3 / 4
    sigma = 0.05
    delta = 0.1

    # Setup
    grid = Grid(L_grid, N_grid)
    potential = infinite_box(grid, L_grid)

    # Initialize the Propagator engine
    print("Diagonalizing Hamiltonian (this may take a few seconds)...")
    start_time = time.time()
    prop = Propagator(grid, potential)
    print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

    # 2. Initialize state |1> (Localized on the RIGHT)
    wf_initial = Wavefunction(grid)
    np.random.seed(42)

    wf_initial.init_gaussian(mu=mu, sigma=sigma)
    # wf_initial.init_sine_wave(mu=mu, delta=delta, n_mode=5)
    # wf_initial.init_random_packet(mu=mu, delta=delta)

    # Calculate theoretical gate times
    E1 = np.pow(np.pi * hbar, 2) / (2 * m * L_grid**2)
    # BUG: For some reason, this works better.
    # E1 = prop.evals[0]
    T_h = (np.pi * hbar) / (2 * E1)
    T_f = 2 * T_h

    # Evolve
    wf_h = Wavefunction(grid)
    wf_f = Wavefunction(grid)

    prop.set_initial_state(wf_initial)
    prop.evolve(T_h, wf_h)
    prop.evolve(T_f, wf_f)

    # Visual validation
    print("\nSimulating...")
    visual_results(
        grid,
        wf_initial,
        prop,
        wf_h=wf_h,
        wf_f=wf_f,
        T_h=T_h,
        T_f=T_f,
    )

    # Animation
    print("\nStarting animation of wavefunction evolution...")
    animate_simulation(
        grid,
        prop,
        wf_initial,
        total_time=T_f * 2,
        T_h=T_h,
        T_f=T_f,
        fps=30,
    )

    # 2D Visualization
    print("\nPlotting 2D waveform evolution...")
    plot_waveform_evolution_2d(
        grid,
        wf_initial,
        prop,
        n_steps=200,
        max_time=T_f * 2,
    )
