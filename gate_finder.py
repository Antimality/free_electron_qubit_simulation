import numpy as np
from scipy.optimize import minimize_scalar


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
