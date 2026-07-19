import time
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


# Import custom modules
from grid import Grid
from potentials import harmonic_oscillator
from wavefunction import Wavefunction
from propagator import Propagator
from visualizations import (
    visual_results,
    animate_simulation,
    plot_waveform_evolution_2d,
    animate_bloch_sphere,
    animate_combined_simulation,
)

if __name__ == "__main__":
    # Constants
    L_grid = 1.0
    N_grid = 1000
    m = 1.0
    hbar = 1.0
    omega = 500.0
    offset = L_grid / 4.0
    sigma = 0.05
    delta = 0.1

    # Setup
    grid = Grid(L_grid, N_grid)
    potential = harmonic_oscillator(grid, m, omega)

    # Initialize the Propagator engine
    print("Diagonalizing Hamiltonian (this may take a few seconds)...")
    start_time = time.time()
    prop = Propagator(grid, potential, m=m, hbar=hbar)
    print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

    # 2. Initialize state |1> (Localized on the RIGHT)
    wf_initial = Wavefunction(grid)
    np.random.seed(42)

    wf_initial.init_coherent_state(offset=offset, omega=omega, m=m, hbar=hbar)
    # wf_initial.init_sine_wave(mu=mu, delta=delta, n_mode=5)
    # wf_initial.init_random_packet(mu=mu, delta=delta)

    # Calculate theoretical gate times
    T_f = (np.pi * hbar) / omega

    # Evolve
    wf_f = Wavefunction(grid)

    prop.set_initial_state(wf_initial)
    prop.evolve(T_f, wf_f)

    # Visual validation
    print("\nSimulating...")
    visual_results(
        grid,
        wf_initial,
        prop,
        wf_f=wf_f,
        T_f=T_f,
    )

       # Combined Split-Screen Animation
    print("\nStarting combined split-screen animation...")
    animate_combined_simulation(
        grid,
        prop,
        wf_initial,
        total_time=T_f * 2,
        T_f=T_f,
        fps=30,
    )

    # Animation
    print("\nStarting animation of wavefunction evolution...")
    animate_simulation(
        grid,
        prop,
        wf_initial,
        total_time=T_f * 2,
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

    #   # Animation on the bloch sphere
    # print("\nStarting animation of bloch sphere ...")
    # animate_bloch_sphere(
    #     grid,
    #     prop,
    #     wf_initial,
    #     total_time=T_f * 2,
    #     T_f=T_f,
    #     fps=30,
    # )
