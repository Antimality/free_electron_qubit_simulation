import time

import numpy as np
from joblib import Memory

# Import custom modules
from grid import Grid
from potentials import complex_absorbing_potential, harmonic_oscillator
from propagator import Propagator
from visualizations import (
    animate_combined_simulation,
    plot_waveform_evolution_2d,
    visual_results,
)
from wavefunction import Wavefunction

memory = Memory(location=".sim_cache", verbose=1)


@memory.cache
def run_simulation():
    # ruff: disable[F841]
    # Constants
    L_grid = 1.0
    N_grid = 1000
    m = 1.0
    hbar = 1.0
    omega = 5.0
    offset = L_grid / 4.0
    sigma = 0.05
    delta = 0.1
    # ruff: enable[F841]

    # Setup
    grid = Grid(L_grid, N_grid, padding_mult=7.0)

    real_potential = harmonic_oscillator(grid, m, omega)
    cap = complex_absorbing_potential(grid, width=0.5, strength=100.0)
    potential = real_potential - 1j * cap

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

    # Theoretical gate time
    T_f = np.pi * hbar / omega

    # Simulated gate time
    # E0 = np.real(prop.evals[0])
    # E1 = np.real(prop.evals[1])
    # delta_E = E1 - E0
    # T_f = (np.pi * hbar) / delta_E

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
        y_lim=1.5,
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


if __name__ == "__main__":
    run_simulation()
