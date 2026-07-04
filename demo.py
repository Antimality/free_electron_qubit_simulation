import time
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from grid import Grid
from potentials import infinite_box, double_gaussian
from wavefunction import Wavefunction
from propagator import Propagator
from gate_finder import find_bit_flip_time, find_hadamard_time
from metrics import get_fidelity, get_ideal_bitflip, get_ideal_hadamard
from visualizations import (
    plot_theory_verifications,
    animate_simulation,
    extract_odd_even_states,
)

if __name__ == "__main__":
    # 1. Setup
    L_grid = 12.0
    omega = 0.5
    mu = 5.0
    N_points = 1000

    grid = Grid(L_grid, N_points)
    potential = double_gaussian(grid, U0=1, sigma=1, mu=10)

    # Initialize the Propagator engine
    print("Diagonalizing Hamiltonian (this may take a few seconds)...")
    start_time = time.time()
    prop = Propagator(grid, potential)
    print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

    # 2. Initialize state |1> (Localized on the RIGHT)
    wf_initial = Wavefunction(grid)
    wf_initial.init_coherent_state(mu=5.0, omega=omega, m=1.0, hbar=1.0)
    # wf_initial.init_sine_wave(xmin=2.0, xmax=8.0, n_mode=2)

    dE = np.abs(prop.evals[0] - prop.evals[1])
    analytic_T_f = np.pi / dE
    analytic_T_h = analytic_T_f / 2

    # 3. Compute Gate Times
    print("\nSearching for exact gate times:")
    print("T_h:")
    start_time = time.time()
    T_h, _ = find_hadamard_time(prop, wf_initial, max_time_search=analytic_T_h * 1.5)
    print(f"T_h found in {time.time() - start_time:.2f} seconds.")
    print("T_f:")
    start_time = time.time()
    T_f, _ = find_bit_flip_time(prop, wf_initial, max_time_search=analytic_T_f * 1.5)
    print(f"T_f found in {time.time() - start_time:.2f} seconds.")

    print(f"Theoretical T_h: {analytic_T_h:.2f}")
    print(f"Calculated T_h: {T_h:.2f}")
    print(f"Theoretical T_f: {analytic_T_f:.2f}")
    print(f"Calculated T_f: {T_f:.2f}")

    # 4. Evolve and calculate Fidelities
    wf_h = Wavefunction(grid)
    wf_h_theo = Wavefunction(grid)
    wf_f = Wavefunction(grid)
    wf_f_theo = Wavefunction(grid)

    prop.set_initial_state(wf_initial)
    prop.evolve(T_h, wf_h)
    prop.evolve(analytic_T_h, wf_h_theo)
    prop.evolve(T_f, wf_f)
    prop.evolve(analytic_T_f, wf_f_theo)

    # Generate ideal targets and calculate scores
    target_f = get_ideal_bitflip(wf_initial)
    target_h = get_ideal_hadamard(wf_initial)

    fidelity_h = get_fidelity(wf_h.psi, target_h, grid.dx)
    fidelity_f = get_fidelity(wf_f.psi, target_f, grid.dx)

    # 5. Plotting Wavefunctions
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    plt.suptitle("Passive Time Evolution from $|1\\rangle$", fontsize=16)

    plot_data = [
        (
            wf_initial.psi,
            wf_initial.psi,
            wf_initial.psi,
            "Initial State $|1\\rangle$\n($t=0$)",
            axs[0],
            None,
        ),
        (
            wf_h.psi,
            wf_h_theo.psi,
            target_h,
            f"Hadamard State\n($t={T_h:.1f}$)",
            axs[1],
            fidelity_h,
        ),
        (
            wf_f.psi,
            wf_f_theo.psi,
            target_f,
            f"Bit-Flip State $|0\\rangle$\n($t={T_f:.1f}$)",
            axs[2],
            fidelity_f,
        ),
    ]

    for psi, psi_theory, psi_target, title, ax, fidelity in plot_data:
        density_target = np.abs(psi_target) ** 2
        ax.plot(grid.x, density_target, color="green", linestyle=":", alpha=0.8)

        density_target = np.abs(psi_theory) ** 2
        ax.plot(grid.x, density_target, color="grey", linestyle="--", alpha=0.8)

        density = np.abs(psi) ** 2
        ax.plot(grid.x, density, color="#003366", lw=2)
        ax.fill_between(grid.x, density, color="#003366", alpha=0.2)

        ax.axvline(0, color="red", linestyle="--", alpha=0.6)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Position (x)")
        ax.set_xlim(-L_grid, L_grid)
        ax.grid(alpha=0.3)

        if fidelity is not None:
            # Display fidelity score in a textbox
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(
                0.05,
                0.95,
                f"Fidelity: {fidelity * 100:.1f}%",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

    axs[0].set_ylabel("Probability Density $|\\psi(x)|^2$")
    plt.tight_layout()
    plt.show()

    # # 6. Plot Theory Verifications (Overlap Matrix & Phase)
    # print("\nGenerating theoretical verification plots...")
    # plot_theory_verifications(prop)

    # # 7. Odd/even
    # print("\nGenerating odd/even plots...")
    # extract_odd_even_states(grid, prop.evecs)

    # # 8. Animation
    print("\nStarting animation of wavefunction evolution...")
    # Animate over 2x the bit-flip time to see the full cycle
    animate_simulation(
        grid,
        prop,
        wf_initial,
        total_time=analytic_T_f * 1.25,
        T_h=T_h,
        T_f=T_f,
        frames=200,
    )
