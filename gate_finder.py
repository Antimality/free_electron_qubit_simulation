import numpy as np
from scipy.optimize import minimize_scalar

from metrics import get_fidelity, get_ideal_hadamard


def global_time_search(objective_func, max_time, points=1000):
    """Performs a coarse grid search to find the global optimum, then refines it."""
    t_evals = np.linspace(1e-2, max_time, points)
    vals = [objective_func(t) for t in t_evals]

    # Find the best time in the coarse array
    best_idx = np.argmin(vals)
    best_t_guess = t_evals[best_idx]

    # Refine locally around that best guess
    window = max_time / points
    res = minimize_scalar(
        objective_func,
        bounds=(max(0, best_t_guess - window), best_t_guess + window),
        method="bounded",
    )
    return res.x, res.fun


def find_hadamard_time(propagator, initial_wf, max_time_search):
    propagator.set_initial_state(initial_wf)
    wf_t = type(initial_wf)(initial_wf.grid)

    def objective_function(t):
        propagator.evolve(t, wf_t)
        p_left = wf_t.get_left_probability()
        return (p_left - 0.5) ** 2

    return global_time_search(objective_function, max_time_search)


def find_bit_flip_time(propagator, initial_wf, max_time_search):
    propagator.set_initial_state(initial_wf)
    wf_t = type(initial_wf)(initial_wf.grid)

    def objective_function(t):
        propagator.evolve(t, wf_t)
        # Starting in |1> (right), so we want to MAXIMIZE probability on the left.
        # Minimizing the negative left probability achieves this.
        return -wf_t.get_left_probability()

    return global_time_search(objective_function, max_time_search)


def find_q1_hadamard_q2_idle(prop1, prop2, wf1_init, wf2_init, max_time, points=2000):
    """
    Searches for a time where Qubit 1 is in a Hadamard state
    AND Qubit 2 is in its initial state.
    """
    t_evals = np.linspace(0.01, max_time, points)

    # Defiene target states
    target_wf1 = get_ideal_hadamard(wf1_init)
    target_wf2 = wf2_init.psi  # Qubit 2 should remain in its initial state

    wf1_t = type(wf1_init)(wf1_init.grid)
    wf2_t = type(wf2_init)(wf2_init.grid)

    best_t = 0
    best_combined_score = -1
    best_f1 = 0
    best_f2 = 0

    # Init propagators
    prop1.set_initial_state(wf1_init)
    prop2.set_initial_state(wf2_init)

    for t in t_evals:
        prop1.evolve(t, wf1_t)
        prop2.evolve(t, wf2_t)

        f1 = get_fidelity(wf1_t.psi, target_wf1, wf1_init.grid.dx)
        f2 = get_fidelity(wf2_t.psi, target_wf2, wf2_init.grid.dx)

        # The combined score is the product of the fidelities (both should be high)
        score = f1 * f2

        if score > best_combined_score:
            best_combined_score = score
            best_t = t
            best_f1 = f1
            best_f2 = f2

    return best_t, best_f1, best_f2
