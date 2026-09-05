import numpy as np
import qutip as qt

from grid import Grid, Grid2D
from potentials import (
    adjacent_double_wells_2d,
    complex_absorbing_potential_2d,
    coulomb_interaction,
    double_gaussian,
)
from propagator import Propagator, SplitOperatorPropagator2D
from visualizations_2d import animate_cnot_circuit_2d, get_local_qutip_qubit
from wavefunction import Wavefunction2D


def apply_virtual_z(wf, center_q1, center_q2, theta_1, theta_2):
    step_1 = 0.5 * (1.0 + np.tanh((wf.grid.x1 - center_q1) / 0.05))
    step_2 = 0.5 * (1.0 + np.tanh((wf.grid.x2 - center_q2) / 0.05))
    phase_mask = theta_1 * step_1 + theta_2 * step_2
    wf.psi *= np.exp(1j * phase_mask)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Intialization and gate calibration
    # ---------------------------------------------------------
    L_grid, N_grid = 1.0, 256
    m, hbar = 1.0, 1.0
    D, mu = 2.0, 0.4
    sigma_well, sigma_init = 0.12, 0.092
    padding_val = 8.0

    U0_hold, U0_tunnel = -200.0, -100.0
    k_e = 5.0

    print("Calibrating Gates...")
    # T_h calibration
    grid1d = Grid(L_grid, N_grid, padding_mult=padding_val)
    V1d_tunnel = double_gaussian(grid1d, U0=U0_tunnel, sigma=sigma_well, mu=mu)
    prop1d = Propagator(grid1d, V1d_tunnel, m=m, hbar=hbar)
    T_h = ((np.pi * hbar) / (np.real(prop1d.evals[1]) - np.real(prop1d.evals[0]))) / 2.0

    # T_cz calibration and Z correction
    E00, E11 = k_e / D, k_e / D
    E01, E10 = k_e / (D + 2 * mu), k_e / (D - 2 * mu)
    T_cz = (np.pi * hbar) / abs(E00 + E11 - E01 - E10)

    theta_1 = (E10 * T_cz / hbar) - (E00 * T_cz / hbar)
    theta_2 = (E01 * T_cz / hbar) - (E00 * T_cz / hbar)

    # ---------------------------------------------------------
    # Sequence timing
    # ---------------------------------------------------------
    T_hold = 0.8

    t_h1_start = T_hold
    t_h1_end = t_h1_start + T_h

    t_cz_start = t_h1_end + T_hold
    t_cz_end = t_cz_start + T_cz

    t_h2_start = t_cz_end + T_hold
    t_h2_end = t_h2_start + T_h

    total_time = t_h2_end + T_hold
    time_schedule = (t_h1_start, t_h1_end, t_cz_start, t_cz_end, t_h2_start, t_h2_end)

    # ---------------------------------------------------------
    # Simulation/Visualization
    # ---------------------------------------------------------
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
    V_cz = V_hold + coulomb_interaction(grid2d, k_e=k_e, softening_a=0.1)

    wf_initial = Wavefunction2D(grid2d)
    center_q1, center_q2 = -D / 2.0, D / 2.0

    # Initialization to |01> state
    wf_initial.init_gaussian(mu1=center_q1 - mu, mu2=center_q2 + mu, sigma=sigma_init)
    current_wf = Wavefunction2D(grid2d)

    prop2d = SplitOperatorPropagator2D(grid2d, V_hold, m=m, hbar=hbar)
    prop2d.set_initial_state(wf_initial)

    # Target state is |11> after CNOT
    target_state_q1 = qt.basis(2, 1)
    target_state_q2 = qt.basis(2, 1)

    fps = 30
    time_array = np.linspace(0, total_time, int(total_time / 0.1))

    marginals_1, marginals_2 = [], []
    bloch_vectors_1, bloch_vectors_2 = [], []
    fidelities_q1, fidelities_q2 = [], []

    z_corrected = False

    print("Pre-computing complete CNOT quantum circuit... Please wait.")
    for t in time_array:
        # Set the correct potential based on the time schedule
        if t_h1_start <= t < t_h1_end:
            prop2d.update_potential(V_hadamard_q1)
        elif t_cz_start <= t < t_cz_end:
            prop2d.update_potential(V_cz)
        elif t_h2_start <= t < t_h2_end:
            prop2d.update_potential(V_hadamard_q1)
        else:
            if t >= t_cz_end and not z_corrected:
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

    print("Launching Integrated Circuit Dashboard...")
    anim, controls = animate_cnot_circuit_2d(
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
        time_schedule,
        fps=fps,
    )
