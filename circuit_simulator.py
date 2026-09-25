import numpy as np
import qutip as qt
from joblib import Memory
from tqdm import tqdm

from gates import GateLibrary
from grid import Grid, Grid2D
from potentials import complex_absorbing_potential_2d
from propagator import SplitOperatorPropagator2D
from visualizations_2d import get_local_qutip_qubit
from wavefunction import Wavefunction2D

# =====================================================================
# GLOBAL HARDWARE & TIMING PARAMETERS
# =====================================================================

PARAMS = {
    # Engine Config
    "L_grid": 1.0,
    "N_grid": 256,
    "padding_val": 8.0,
    "m": 1.0,
    "hbar": 1.0,
    # Spatial Qubit Topology
    "D": 2.0,
    "mu": 0.4,
    "sigma_well": 0.12,
    "sigma_init": 0.092,
    # Potentials
    "U0_hold": -200.0,
    "U0_tunnel": -100.0,
    "k_e": 5.0,
    "softening_a": 0.1,
    "dU": 10.0,
    "cz_slowdown_mult": 15.0,
    # Sequence Timing Definitions
    "T_hold": 0.8,
    "dt_sim": 0.005,
    "dt_anim": 0.01,
    "fps": 30,
}


class Simulator:
    """Executes a defined sequence of logical quantum gates."""

    def __init__(self, p, initial_state, target_state, logical_sequence):
        self.params = p
        self.grid1d = Grid(p["L_grid"], p["N_grid"], padding_mult=p["padding_val"])
        self.grid2d = Grid2D(p["L_grid"], p["N_grid"], padding_mult=p["padding_val"])
        self.abso = complex_absorbing_potential_2d(
            self.grid2d, width=0.2, strength=100.0
        )

        print("Calibrating Gate Library...")
        self.gate_lib = GateLibrary(p, self.grid1d, self.grid2d, self.abso)

        self.initial_state = initial_state
        self.target_state = target_state
        self._build_circuit(logical_sequence)

    def _build_circuit(self, logical_sequence):
        """Concatenates individual gate components into a master timeline."""
        self.sequence = []
        current_time = 0.0

        for gate_name in logical_sequence:
            sub_sequence = self.gate_lib.get_sequence(gate_name)
            for step_name, duration, potential in sub_sequence:
                self.sequence.append((step_name, duration, potential))
                current_time += duration

        self.total_time = current_time
        self.gate_schedule = self._extract_gate_schedule(logical_sequence)

    def _extract_gate_schedule(self, logical_sequence):
        """Extracts dynamic start and end times for all active gates."""
        schedule = []
        current_time = 0.0

        for gate_name in logical_sequence:
            gate_duration = sum(
                dur for _, dur, _ in self.gate_lib.get_sequence(gate_name)
            )

            if gate_name != "HOLD":
                schedule.append(
                    {
                        "name": gate_name,
                        "start": current_time,
                        "end": current_time + gate_duration,
                    }
                )

            current_time += gate_duration

        return schedule

    def get_active_potential(self, t):
        """Routes the timeline: returns the gate name and potential active at time t."""
        current_time = 0.0
        for name, duration, potential in self.sequence:
            current_time += duration
            if t <= current_time:
                return name, potential
        return self.sequence[-1][0], self.sequence[-1][2]

    def run(self):
        """Executes the split-step evolution using the state-tracked sequence."""
        print("Executing quantum circuit simulation...")
        p = self.params
        wf = Wavefunction2D(self.grid2d)
        center_q1, center_q2 = -p["D"] / 2.0, p["D"] / 2.0

        # --- 1. Dynamic Initialization ---
        wf.init_logical_state(
            self.initial_state[0],
            self.initial_state[1],
            center_q1,
            center_q2,
            p["mu"],
            p["sigma_init"],
        )

        prop2d = SplitOperatorPropagator2D(
            self.grid2d, self.gate_lib.V_hold, m=p["m"], hbar=p["hbar"]
        )
        prop2d.set_initial_state(wf)

        # --- 2. Dynamic Target State Parsing ---
        def parse_qutip_state(state_val):
            """Maps the string state to the corresponding QuTiP vector."""
            s = str(state_val)
            if s == "0":
                return qt.basis(2, 0)
            elif s == "1":
                return qt.basis(2, 1)
            elif s == "+":
                return (qt.basis(2, 0) + qt.basis(2, 1)).unit()
            elif s == "-":
                return (qt.basis(2, 0) - qt.basis(2, 1)).unit()
            raise ValueError("State must be '0', '1', '+', or '-'")

        target_state_q1 = parse_qutip_state(self.target_state[0])
        target_state_q2 = parse_qutip_state(self.target_state[1])

        time_array = np.linspace(
            0, self.total_time, int(self.total_time / p["dt_anim"])
        )

        marginals_1, marginals_2 = [], []
        bloch_1, bloch_2 = [], []
        active_gate = None

        for t in tqdm(time_array, desc="Simulating", ncols=100, unit="step"):
            current_gate, V_active = self.get_active_potential(t)

            if current_gate != active_gate:
                prop2d.update_potential(V_active)
                active_gate = current_gate

            prop2d.evolve(t, wf, dt=p["dt_sim"])

            density_2d = np.abs(wf.psi) ** 2
            marginals_1.append(np.sum(density_2d, axis=1) * self.grid2d.dx)
            marginals_2.append(np.sum(density_2d, axis=0) * self.grid2d.dx)

            state_2q = get_local_qutip_qubit(wf, center_q1, center_q2)
            rho_2q = state_2q * state_2q.dag()
            rho_1, rho_2 = rho_2q.ptrace(0), rho_2q.ptrace(1)

            bloch_1.append(
                [
                    qt.expect(qt.sigmax(), rho_1),
                    qt.expect(qt.sigmay(), rho_1),
                    qt.expect(qt.sigmaz(), rho_1),
                ]
            )
            bloch_2.append(
                [
                    qt.expect(qt.sigmax(), rho_2),
                    qt.expect(qt.sigmay(), rho_2),
                    qt.expect(qt.sigmaz(), rho_2),
                ]
            )

        fid_q1 = qt.fidelity(rho_1, target_state_q1) ** 2
        fid_q2 = qt.fidelity(rho_2, target_state_q2) ** 2

        return {
            "time": time_array,
            "m1": marginals_1,
            "m2": marginals_2,
            "b1": bloch_1,
            "b2": bloch_2,
            "fid_q1": fid_q1,
            "fid_q2": fid_q2,
        }


# Initialize Joblib Memory cache
memory = Memory(location=".sim_cache", verbose=1)


@memory.cache
def run_simulation(p, initial_state, target_state, sequence):
    """
    Wrapper function monitored by Joblib. If the inputs match a previous run,
    it instantly loads the returned arrays from disk instead of re-running the physics.
    """
    sim = Simulator(p, initial_state, target_state, sequence)
    res = sim.run()
    return res, sim.grid2d, sim.gate_schedule
