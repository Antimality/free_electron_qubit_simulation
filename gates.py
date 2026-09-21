import numpy as np

from potentials import (
    adjacent_double_wells_2d,
    coulomb_interaction,
    double_gaussian,
)
from propagator import Propagator


class GateLibrary:
    """Pre-computes and serves physical potential matrices for quantum gates."""

    def __init__(self, p, grid1d, grid2d, abso):
        self.p = p
        self.grid1d = grid1d
        self.grid2d = grid2d
        self.abso = abso

        self._calibrate_timings()
        self._build_gate_potentials()

    def _calibrate_timings(self):
        """Calculates exact analytical timings required for the quantum gates."""
        p = self.p

        # ?: Is it really necessary to calibrate T_h like this and not properly calculate it?
        # T_h (Tunneling/Hadamard) Calibration
        V1d_tunnel = double_gaussian(
            self.grid1d, U0=p["U0_tunnel"], sigma=p["sigma_well"], mu=p["mu"]
        )
        prop1d = Propagator(self.grid1d, V1d_tunnel, m=p["m"], hbar=p["hbar"])
        self.T_h = (
            (np.pi * p["hbar"]) / (np.real(prop1d.evals[1]) - np.real(prop1d.evals[0]))
        ) / 2.0

        # CZ Interaction Energy Calibration
        E00, E11 = p["k_e"] / p["D"], p["k_e"] / p["D"]
        E01 = p["k_e"] / (p["D"] + 2 * p["mu"])
        E10 = p["k_e"] / (p["D"] - 2 * p["mu"])

        self.T_cz = (np.pi * p["hbar"]) / abs(E00 + E11 - E01 - E10)
        self.T_phase = (np.pi * p["hbar"]) / (2.0 * p["dU"])
        self.T_z_corr = self.T_phase

        # Virtual Z correction phases derived from the CZ interaction
        self.theta_1 = (E10 * self.T_cz / p["hbar"]) - (E00 * self.T_cz / p["hbar"])
        self.theta_2 = (E01 * self.T_cz / p["hbar"]) - (E00 * self.T_cz / p["hbar"])

    def _build_gate_potentials(self):
        """Pre-generates all 2D potential matrices used by the gates."""
        p = self.p

        self.V_hold = (
            adjacent_double_wells_2d(
                self.grid2d,
                U0_1=p["U0_hold"],
                U0_2=p["U0_hold"],
                sigma=p["sigma_well"],
                mu=p["mu"],
                D=p["D"],
            )
            - 1j * self.abso
        )

        # --- Qubit 1 (Target) Potentials ---
        self.V_hadamard_q1 = (
            adjacent_double_wells_2d(
                self.grid2d,
                U0_1=p["U0_tunnel"],
                U0_2=p["U0_hold"],
                sigma=p["sigma_well"],
                mu=p["mu"],
                D=p["D"],
            )
            - 1j * self.abso
        )

        self.V_phase_shift_q1 = (
            adjacent_double_wells_2d(
                self.grid2d,
                U0_1=p["U0_hold"],
                U0_2=p["U0_hold"],
                sigma=p["sigma_well"],
                mu=p["mu"],
                D=p["D"],
                dU1=p["dU"],
                dU2=0,
            )
            - 1j * self.abso
        )

        # --- Qubit 2 (Target) Potentials ---
        self.V_hadamard_q2 = (
            adjacent_double_wells_2d(
                self.grid2d,
                U0_1=p["U0_hold"],
                U0_2=p["U0_tunnel"],
                sigma=p["sigma_well"],
                mu=p["mu"],
                D=p["D"],
            )
            - 1j * self.abso
        )

        self.V_phase_shift_q2 = (
            adjacent_double_wells_2d(
                self.grid2d,
                U0_1=p["U0_hold"],
                U0_2=p["U0_hold"],
                sigma=p["sigma_well"],
                mu=p["mu"],
                D=p["D"],
                dU1=0,
                dU2=p["dU"],
            )
            - 1j * self.abso
        )

        # --- Entangling Potentials ---
        self.V_cz = self.V_hold + coulomb_interaction(
            self.grid2d, k_e=p["k_e"], softening_a=0.1
        )

        dU1_z = -(self.theta_1 * p["hbar"]) / self.T_z_corr
        dU2_z = -(self.theta_2 * p["hbar"]) / self.T_z_corr

        self.V_z_corr = (
            adjacent_double_wells_2d(
                self.grid2d,
                U0_1=p["U0_hold"],
                U0_2=p["U0_hold"],
                sigma=p["sigma_well"],
                mu=p["mu"],
                D=p["D"],
                dU1=dU1_z,
                dU2=dU2_z,
            )
            - 1j * self.abso
        )

    def get_sequence(self, gate_name):
        """Returns the physical breakdown of a logical gate as (Name, Duration, Potential)."""
        p = self.p
        if gate_name == "HOLD":
            return [("HOLD", p["T_hold"], self.V_hold)]

        elif gate_name == "H1":
            return [
                ("H1_PHASE", self.T_phase, self.V_phase_shift_q1),
                ("H1_TUNNEL", self.T_h, self.V_hadamard_q1),
                ("H1_PHASE", self.T_phase, self.V_phase_shift_q1),
            ]

        elif gate_name == "H2":
            return [
                ("H2_PHASE", self.T_phase, self.V_phase_shift_q2),
                ("H2_TUNNEL", self.T_h, self.V_hadamard_q2),
                ("H2_PHASE", self.T_phase, self.V_phase_shift_q2),
            ]

        elif gate_name == "CZ":
            # Restoring the necessary HOLD period allows the wavepacket to
            # recover from breathing before the Gaussian phase mask is applied.
            return [
                ("CZ_INTERACT", self.T_cz, self.V_cz),
                ("HOLD", p["T_hold"], self.V_hold),
                ("Z_CORR", self.T_z_corr, self.V_z_corr),
            ]

        else:
            raise ValueError(f"Unknown gate requested: {gate_name}")
