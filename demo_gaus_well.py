import time

import numpy as np

from grid import Grid
from potentials import gaussian_walls
from propagator import Propagator
from visualizations import (
    animate_simulation,
    plot_waveform_evolution_2d,
    visual_results,
)
from wavefunction import Wavefunction

if __name__ == "__main__":
    # 1. Setup Grid
    # Spanning slightly beyond the walls (e.g., L_grid = 2.0) prevents the wavepacket
    # tails from violently clipping the numerical boundaries.
    L_grid = 1.0
    N_points = 1000
    grid = Grid(L_grid, N_points, padding_mult=2.0)

    # 2. Setup the Gaussian Walls
    U0 = 10000.0  # High barrier to approximate an infinite well
    sigma = 0.005  # Steep enough to leave the center flat
    potential = gaussian_walls(grid, U0, sigma, 0.5)

    # 3. Initialize the Propagator engine
    m = 1.0
    hbar = 1.0
    print("Diagonalizing Hamiltonian (this may take a few seconds)...")
    start_time = time.time()
    prop = Propagator(grid, potential, hbar=hbar)
    print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

    # 4. Initialize state |0> as a Coherent State
    # We place the wavepacket halfway between the left wall and the symmetry axis
    wf_initial = Wavefunction(grid)
    # omega = 40.0
    # mu_left = (grid.L / 2.0) - 0.25
    # wf_initial.init_coherent_state(offset=mu_left, omega=omega, m=m, hbar=hbar)
    wf_initial.init_gaussian(mu=0.25, sigma=0.05)

    # String example with double backslash escaping for LaTeX
    plot_title = "Wavepacket Probability Density \\|\\psi(x)\\|^2"

    # 5. Calculate theoretical gate times using the exact energy gap
    dE = (np.pi * hbar) ** 2 / (2 * m * L_grid**2)
    # dE = np.abs(prop.evals[0] - prop.evals[1])
    T_h = (np.pi * hbar) / (2 * dE)
    T_f = 2 * T_h
    print(f"Calculated gate times: T_h = {T_h:.4f}, T_f = {T_f:.4f}")

    # 7. Evolve to T_f
    wf_f = Wavefunction(grid)
    prop.set_initial_state(wf_initial)
    prop.evolve(T_f, wf_f)

    # 8. Visualizations
    print("\nSimulating and generating visual results...")
    visual_results(
        grid,
        wf_initial,
        prop,
        wf_f=wf_f,
        T_f=T_f,
    )

    print("\nStarting animation of wavefunction evolution...")
    animate_simulation(
        grid,
        prop,
        wf_initial,
        total_time=T_f * 2,
        T_h=T_h,
        T_f=T_f,
        fps=30,
        y_lim=10,
    )

    print("\nPlotting 2D waveform evolution...")
    plot_waveform_evolution_2d(
        grid,
        wf_initial,
        prop,
        n_steps=200,
        max_time=T_f * 2,
        cmap="magma",
    )
