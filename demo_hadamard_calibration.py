import numpy as np
import qutip as qt

from grid import Grid, Grid2D
from potentials import (
    adjacent_double_wells_2d,
    complex_absorbing_potential_2d,
    double_gaussian,
)
from propagator import Propagator, SplitOperatorPropagator2D
from visualizations_2d import (
    animate_combined_simulation_2d,
    get_local_qutip_qubit,
    visual_results_2d,
)
from wavefunction import Wavefunction2D

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
    U0_tunnel = -100.0

    dU = 10.0 #different potential between the wells for phase shift

    # --- Step 1: Calibration ---
    print("Calibrating exact Hadamard time...")
    grid1d = Grid(L_grid, N_grid, padding_mult=padding_val)
    V1d_tunnel = double_gaussian(grid1d, U0=U0_tunnel, sigma=sigma_well, mu=mu)

    prop1d = Propagator(grid1d, V1d_tunnel, m=m, hbar=hbar)
    E0 = np.real(prop1d.evals[0])
    E1 = np.real(prop1d.evals[1])

    T_h = ((np.pi * hbar) / (E1 - E0)) / 2.0
    print(f"Calculated Exact Hadamard Time (T_h): {T_h:.4f}")

    # --- Step 2: 2D Setup ---
    grid2d = Grid2D(L_grid, N_grid, padding_mult=padding_val)
    abso = complex_absorbing_potential_2d(grid2d, width=0.2, strength=100.0)

    V_hold = (
        adjacent_double_wells_2d(
            grid2d, U0_1=U0_hold, U0_2=U0_hold, sigma=sigma_well, mu=mu, D=D
        )
        - 1j * abso
    )
    V_hadamard_q1 = (
        adjacent_double_wells_2d(
            grid2d, U0_1=U0_tunnel, U0_2=U0_hold, sigma=sigma_well, mu=mu, D=D
        )
        - 1j * abso
    )
    
    V_phase_shift = (
        adjacent_double_wells_2d(
            grid2d, U0_1=U0_hold, U0_2=U0_hold , sigma=sigma_well, mu=mu, D=D, dU1 = dU
        )
        - 1j * abso
    )

    wf_initial = Wavefunction2D(grid2d)
    center_q1, center_q2 = -D / 2.0, D / 2.0
    wf_initial.init_gaussian(mu1=center_q1 - mu, mu2=center_q2 - mu, sigma=sigma_init)

    prop2d = SplitOperatorPropagator2D(grid2d, V_hold, m=m, hbar=hbar)
    prop2d.set_initial_state(wf_initial)
    current_wf = Wavefunction2D(grid2d)

    target_state_q1 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    target_state_q2 = qt.basis(2, 0)

    # --- Step 3: Pre-Compute Evolution ---
    print("Pre-computing quantum evolution... Please wait.")
    T_hold_start = 1.0
    T_hold_end = 0.0
    T_phase = (np.pi * hbar) / (2.0 * dU)
    total_time = T_hold_start + T_h + T_phase + T_hold_end


    fps = 30
    dt_anim = 0.01
    time_steps = int(total_time / dt_anim)
    time_array = np.linspace(0, total_time, time_steps)

    marginals_1, marginals_2 = [], []
    bloch_vectors_1, bloch_vectors_2 = [], []
    fidelities_q1, fidelities_q2 = [], []

    for t in time_array:
        if t <= T_hold_start:
            prop2d.update_potential(V_hold)
        elif t <= T_hold_start + T_h:
            prop2d.update_potential(V_hadamard_q1)
        elif t <= T_hold_start + T_h + T_phase:
            prop2d.update_potential(V_phase_shift)
        else:
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
        T_h,
        fps=fps,
        gate_name="HADAMARD",
        gate_color="#E63946",
        target_labels=("|+>", "|0>"),
        title_str="Interactive Hadamard Gate Evolution",
        jump_btn_text="Jump to Hadamard",
    )
