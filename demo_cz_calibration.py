import numpy as np
import qutip as qt

from grid import Grid2D
from potentials import (
    adjacent_double_wells_2d,
    complex_absorbing_potential_2d,
    coulomb_interaction,
)
from propagator import SplitOperatorPropagator2D
from visualizations_2d import (
    animate_combined_simulation_2d,
    get_local_qutip_qubit,
    visual_results_2d,
)
from wavefunction import Wavefunction2D


def init_cz_state(wf, center_q1, center_q2, mu, sigma):
    """
    Initializes the wavefunction for a two-qubit system in the |+>|0> state.
    """
    x1 = wf.grid.x1
    x2 = wf.grid.x2

    g0_q1 = np.exp(-((x1 - (center_q1 - mu)) ** 2) / (4 * sigma**2))
    g1_q1 = np.exp(-((x1 - (center_q1 + mu)) ** 2) / (4 * sigma**2))
    psi_q1 = (g0_q1 + g1_q1) / np.sqrt(2.0)

    psi_q2 = np.exp(-((x2 - (center_q2 + mu)) ** 2) / (4 * sigma**2))

    wf.psi = (psi_q1 * psi_q2).astype(complex)
    wf.psi /= np.sqrt(np.sum(np.abs(wf.psi) ** 2) * wf.grid.dx**2)


def apply_virtual_z(wf, center_q1, center_q2, theta_1, theta_2):
    """
    Applies virtual Z corrections to the wavefunction for a two-qubit system.
    This is done by applying phase shifts to the wavefunction based on the qubit states.
    """
    # Step functions for each qubit (0 on the left, 1 on the right)
    step_1 = 0.5 * (1.0 + np.tanh((wf.grid.x1 - center_q1) / 0.05))
    step_2 = 0.5 * (1.0 + np.tanh((wf.grid.x2 - center_q2) / 0.05))

    # Calculate the total phase mask and multiply it with the wavefunction
    phase_mask = theta_1 * step_1 + theta_2 * step_2
    wf.psi *= np.exp(1j * phase_mask)


if __name__ == "__main__":
    L_grid = 1.0
    N_grid = 256
    m = 1.0
    hbar = 1.0

    D = 2.0
    mu = 0.4
    sigma_well = 0.12
    sigma_init = 0.092
    padding_val = 8.0

    U0_hold = -200.0
    k_e = 5.0

    # --- Calibration of Universal CZ Time & Phase Corrections ---
    print("Calibrating exact CZ time and Virtual Z corrections...")
    E00 = k_e / D
    E11 = k_e / D
    E01 = k_e / (D + 2 * mu)
    E10 = k_e / (D - 2 * mu)

    E_int = E00 + E11 - E01 - E10
    T_cz = (np.pi * hbar) / abs(E_int)

    # Calculate the phase shifts that occured for each quibit during the CZ gate
    phi_00 = E00 * T_cz / hbar
    phi_01 = E01 * T_cz / hbar
    phi_10 = E10 * T_cz / hbar

    # Calculate the corrections needed
    theta_1 = phi_10 - phi_00
    theta_2 = phi_01 - phi_00

    print(f"Interaction Energy (E_int): {E_int:.4f}")
    print(f"Calculated CZ Time (T_cz): {T_cz:.4f}")
    print(f"Virtual Z Correction Q1: {np.degrees(theta_1):.1f} deg")
    print(f"Virtual Z Correction Q2: {np.degrees(theta_2):.1f} deg")

    # --- Step 2: 2D Setup ---
    grid2d = Grid2D(L_grid, N_grid, padding_mult=padding_val)
    abso = complex_absorbing_potential_2d(grid2d, width=0.2, strength=100.0)

    V_hold = (
        adjacent_double_wells_2d(
            grid2d, U0_1=U0_hold, U0_2=U0_hold, sigma=sigma_well, mu=mu, D=D
        )
        - 1j * abso
    )
    V_coulomb = coulomb_interaction(grid2d, k_e=k_e, softening_a=0.1)
    V_cz = V_hold + V_coulomb

    wf_initial = Wavefunction2D(grid2d)
    center_q1, center_q2 = -D / 2.0, D / 2.0

    init_cz_state(wf_initial, center_q1, center_q2, mu, sigma_init)

    prop2d = SplitOperatorPropagator2D(grid2d, V_hold, m=m, hbar=hbar)
    prop2d.set_initial_state(wf_initial)
    current_wf = Wavefunction2D(grid2d)

    # Target state is |->|1> after CZ + corrections
    target_state_q1 = (
        qt.basis(2, 0) - qt.basis(2, 1)
    ).unit()  # State |-> (Pointed at -X)
    target_state_q2 = qt.basis(2, 1)  # State |1>

    # --- Step 3: Pre-Compute Evolution ---
    print("Pre-computing quantum evolution... Please wait.")
    T_hold_start = 1.0
    T_hold_end = 1.0
    total_time = T_hold_start + T_cz + T_hold_end

    fps = 30
    dt_anim = 0.1
    time_steps = int(total_time / dt_anim)
    time_array = np.linspace(0, total_time, time_steps)

    marginals_1, marginals_2 = [], []
    bloch_vectors_1, bloch_vectors_2 = [], []
    fidelities_q1, fidelities_q2 = [], []

    z_corrected = False

    for t in time_array:
        if t < T_hold_start:
            prop2d.update_potential(V_hold)
        elif t < T_hold_start + T_cz:
            prop2d.update_potential(V_cz)
        else:
            # Phase correction is applied (exactly) once after CZ gate
            if not z_corrected:
                apply_virtual_z(current_wf, center_q1, center_q2, theta_1, theta_2)
                z_corrected = True
            prop2d.update_potential(V_hold)

        prop2d.evolve(t, current_wf, dt=0.005)

        density_2d = np.abs(current_wf.psi) ** 2
        marginals_1.append(np.sum(density_2d, axis=1) * grid2d.dx)
        marginals_2.append(np.sum(density_2d, axis=0) * grid2d.dx)

        state_2q = get_local_qutip_qubit(current_wf, center_q1, center_q2)
        rho_2q = state_2q * state_2q.dag()
        rho_1, rho_2 = rho_2q.ptrace(0), rho_2q.ptrace(1)

        bloch_vectors_1.append(
            [
                qt.expect(qt.sigmax(), rho_1),
                qt.expect(qt.sigmay(), rho_1),
                qt.expect(qt.sigmaz(), rho_1),
            ]
        )
        bloch_vectors_2.append(
            [
                qt.expect(qt.sigmax(), rho_2),
                qt.expect(qt.sigmay(), rho_2),
                qt.expect(qt.sigmaz(), rho_2),
            ]
        )

        fidelities_q1.append(qt.fidelity(rho_1, target_state_q1) ** 2)
        fidelities_q2.append(qt.fidelity(rho_2, target_state_q2) ** 2)

    # --- Step 4: Visualizations ---
    print("Displaying Final Snapshot... (Close window to open Interactive Player)")
    visual_results_2d(
        current_wf,
        center_q1,
        center_q2,
        mu,
        sigma_well,
        D,
        (fidelities_q1[-1], fidelities_q2[-1]),
        target_labels=("|->", "|1>"),
        title_str="Universal CZ Gate Snapshot (Phase Corrected)",
    )

    print("Launching Interactive Player...")
    anim, controls = animate_combined_simulation_2d(
        grid2d,
        time_array,
        marginals_1,
        marginals_2,
        bloch_vectors_1,
        bloch_vectors_2,
        fidelities_q1,
        fidelities_q2,
        center_q1,
        center_q2,
        mu,
        sigma_well,
        D,
        T_hold_start,
        T_cz,
        fps=fps,
        gate_name="CZ + Virtual Z",
        gate_color="purple",
        target_labels=("|->", "|1>"),
        title_str="Interactive Universal CZ Evolution",
        jump_btn_text="Jump to CZ",
    )
