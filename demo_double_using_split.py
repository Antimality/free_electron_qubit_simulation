import time

import numpy as np

# Import custom modules
from grid import Grid
from potentials import complex_absorbing_potential, double_gaussian
from propagator import Propagator, SplitOperatorPropagator
from visualizations import (
    animate_combined_simulation,
    visual_results,
)
from wavefunction import Wavefunction

if __name__ == "__main__":
    # Constants
    L_grid = 1.0
    N_grid = 1000
    m = 1.0
    hbar = 1.0
    mu = 0.4
    sigma = 0.12
    U0 = -100.0

# Setup
grid = Grid(L_grid, N_grid, 7.0)
real_potential = double_gaussian(grid, U0=U0, sigma=sigma, mu=mu)
abso = complex_absorbing_potential(grid, width=0.2, strength=100.0)
potential = real_potential - 1j * abso


# step 1: compute the states and the gate times
print("Diagonalizing Hamiltonian for static analysis...")
start_time = time.time()
static_prop = Propagator(grid, potential, m=m, hbar=hbar)
print(f"Diagonalization complete in {time.time() - start_time:.2f} seconds.")

E0 = np.real(static_prop.evals[0])
E1 = np.real(static_prop.evals[1])
delta_E = E1 - E0
T_f = (np.pi * hbar) / delta_E
T_h = T_f / 2.0

print(f"Energy Gap (Delta E): {delta_E:.6f}")
print(f"Calculated Bit-Flip Time (T_f): {T_f:.4f}")

psi_0 = static_prop.evecs[:, 0].copy()
psi_1 = static_prop.evecs[:, 1].copy()

idx_right = np.argmin(np.abs(grid.x - mu))
if np.real(psi_0[idx_right]) < 0:
    psi_0 = -psi_0
if np.real(psi_1[idx_right]) < 0:
    psi_1 = -psi_1

localized_psi = (psi_0 + psi_1) / np.sqrt(2.0)
norm = np.sqrt(np.sum(np.abs(localized_psi) ** 2) * grid.dx)

wf_initial = Wavefunction(grid)
wf_initial.psi = localized_psi / norm

# step 2: dainamic evolution using the Split-Operator method
print("\nInitializing Split-Operator engine for time dynamics...")
dyn_prop = SplitOperatorPropagator(grid, potential, m=m, hbar=hbar)
dyn_prop.set_initial_state(wf_initial)

dyn_prop.evecs = static_prop.evecs
wf_h = Wavefunction(grid)
wf_f = Wavefunction(grid)

print("\nEvolving wavefunction...")

dyn_prop.evolve(T_h, wf_h, dt=0.001)
dyn_prop.evolve(T_f, wf_f, dt=0.001)

# step 3:  results visualization
print("\nSimulating results...")
visual_results(
    grid,
    wf_initial,
    dyn_prop,
    wf_h=wf_h,
    wf_f=wf_f,
    T_h=T_h,
    T_f=T_f,
)

print("\nStarting combined split-screen animation...")
animate_combined_simulation(
    grid,
    dyn_prop,
    wf_initial,
    total_time=T_f * 2,
    T_h=T_h,
    T_f=T_f,
    fps=30,
    y_lim=(-5.0, 6.0),
)
