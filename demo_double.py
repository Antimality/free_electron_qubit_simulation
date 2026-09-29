import time

import numpy as np
from joblib import Memory

# Import custom modules
from grid import Grid
from potentials import complex_absorbing_potential, double_gaussian
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
    mu = 0.4
    sigma = 0.12
    U0 = -100.0
    delta = 0.1
    # ruff: enable[F841]

    # Setup
    grid = Grid(L_grid, N_grid, 7.0)
    real_potential = double_gaussian(grid, U0=U0, sigma=sigma, mu=mu)
    abso = complex_absorbing_potential(grid, width=0.2, strength=100.0)
    potential = real_potential - 1j * abso

    # Initialize the Propagator engine
    print("Diagonalizing Hamiltonian (this may take a few seconds)...")
    start_time = time.time()
    prop = Propagator(grid, potential)
    print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

    # 2. Initialize state |1> (Localized on the RIGHT)
    wf_initial = Wavefunction(grid)
    np.random.seed(42)
    psi_0 = prop.evecs[:, 0].copy()
    psi_1 = prop.evecs[:, 1].copy()

    idx_right = np.argmin(np.abs(grid.x - mu))
    if np.real(psi_0[idx_right]) < 0:
        psi_0 = -psi_0
    if np.real(psi_1[idx_right]) < 0:
        psi_1 = -psi_1

    localized_psi = (psi_0 + psi_1) / np.sqrt(2.0)

    norm = np.sqrt(np.sum(np.abs(localized_psi) ** 2) * grid.dx)
    wf_initial.psi = localized_psi / norm

    # wf_initial.init_sine_wave(mu=mu, delta=delta, n_mode=5)
    # wf_initial.init_random_packet(mu=mu, delta=delta)

    # Calculate physical gate times from the energy gap
    E0 = np.real(prop.evals[0])
    E1 = np.real(prop.evals[1])
    delta_E = E1 - E0

    # T_f is the bit-flip time (time for a full Pi phase accumulation)
    T_f = (np.pi * hbar) / delta_E
    T_h = T_f / 2.0  # Hadamard is half the bit-flip time

    # Evolve
    wf_h = Wavefunction(grid)
    wf_f = Wavefunction(grid)

    prop.set_initial_state(wf_initial)
    prop.evolve(T_h, wf_h)
    prop.evolve(T_f, wf_f)

    print(f"Grid range: min={grid.x[0]}, max={grid.x[-1]}")
    print(f"Wavefunction norm at t=0: {np.sum(np.abs(wf_initial.psi) ** 2) * grid.dx}")
    print(f"Potential min/max: {np.min(potential)}, {np.max(potential)}")

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

    # Combined Split-Screen Animation
    print("\nStarting combined split-screen animation...")
    animate_combined_simulation(
        grid,
        prop,
        wf_initial,
        total_time=T_f * 2,
        T_h=T_h,
        T_f=T_f,
        fps=30,
        y_lim=(-5.0, 6.0),
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
