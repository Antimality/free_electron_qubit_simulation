import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from wavefunction import Wavefunction
from metrics import get_fidelity, get_ideal_bitflip, get_ideal_hadamard
from gate_finder import find_bit_flip_time, find_hadamard_time


def visual_results(
    grid,
    wf_initial,
    prop,
    wf_h=None,
    wf_f=None,
    T_h=None,
    T_f=None,
    max_search_time=1000.0,
):
    # Make sure at least one waveform is being displayed
    assert (wf_h is not None) or (wf_f is not None)

    # Generate ideal targets and calculate scores
    if wf_h is not None:
        target_h = get_ideal_hadamard(wf_initial)
        fidelity_h = get_fidelity(wf_h.psi, target_h, grid.dx)

    if wf_f is not None:
        target_f = get_ideal_bitflip(wf_initial)
        fidelity_f = get_fidelity(wf_f.psi, target_f, grid.dx)

    # If times are not provided, find them manually
    if T_h is None:
        T_h, _ = find_hadamard_time(prop, wf_initial, max_time_search=max_search_time)

    if T_f is None:
        T_f, _ = find_bit_flip_time(prop, wf_initial, max_time_search=max_search_time)

    # Plotting Wavefunctions
    subplots = 1 + (wf_h is not None) + (wf_f is not None)
    fig, axs = plt.subplots(1, subplots, figsize=(18, 5), sharey=True)
    plt.suptitle("Passive Time Evolution from $|1\\rangle$", fontsize=16)

    # Prepare the data for plotting
    plot_data = [
        (
            wf_initial.psi,
            wf_initial.psi,
            "Initial State $|1\\rangle$\n($t=0$)",
            axs[0],
            None,
        ),
    ]

    ax_idx = 1
    if wf_h is not None:
        plot_data.append(
            (
                wf_h.psi,
                target_h,
                f"Hadamard State $(|0\\rangle + i|1\\rangle) / \\sqrt{'{2}'}$\n($t={T_h:.1f}$)",
                axs[ax_idx],
                fidelity_h,
            )
        )
        ax_idx += 1

    if wf_f is not None:
        plot_data.append(
            (
                wf_f.psi,
                target_f,
                f"Flipped state $|0\\rangle$\n($t={T_f:.1f}$)",
                axs[ax_idx],
                fidelity_f,
            )
        )

    # Plot
    for psi, psi_target, title, ax, fidelity in plot_data:
        # Target
        density_target = np.abs(psi_target) ** 2
        ax.plot(grid.x, density_target, color="green", linestyle=":", alpha=0.8)

        # Simulated
        density = np.abs(psi) ** 2
        ax.plot(grid.x, density, color="#003366", lw=2)
        ax.fill_between(grid.x, density, color="#003366", alpha=0.2)

        # Symmetry line
        ax.axvline(grid.L / 2, color="red", linestyle="--", alpha=0.6)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Position (x)")
        ax.set_xlim(0, grid.L)
        ax.grid(alpha=0.3)

        # Display fidelity score
        if fidelity is not None:
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

    # Set y-label once for the whole thing
    axs[0].set_ylabel("Probability Density $|\\psi(x)|^2$")
    plt.tight_layout()
    plt.show()


def animate_simulation(
    grid, prop, initial_wf, total_time, T_h=None, T_f=None, fps=30, y_lim=None
):
    """
    Creates an animation of the wavefunction density over time without the fill region.
    Includes a 1-second pause at any integer multiple of T_h and T_f.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 1. Base Plot Configuration ---
    ax.set_xlim(0, grid.L)
    if y_lim is not None:
        ax.set_ylim(0, y_lim)
    ax.set_xlabel("Position (x)", fontsize=12)
    ax.set_ylabel(r"Probability Density $|\psi(x,t)|^2$", fontsize=12)
    ax.set_title("Time Evolution of the Location Qubit", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    # Red dashed symmetry separator line at x=0
    ax.axvline(grid.L / 2, color="red", linestyle="--", alpha=0.6, lw=1.5)

    # Visualizing the potential
    ax.plot(grid.x, prop.potential, color="black", linestyle="-", lw=2, alpha=0.7)

    # Initialize plot elements (line + text)
    dummy_y = np.zeros_like(grid.x)
    (line,) = ax.plot(grid.x, dummy_y, lw=2, color="#003366")
    time_text = ax.text(
        0.05,
        0.92,
        "",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    if T_f is not None:
        # Static overlays mapping the base quantum computational registers
        density_state_1 = initial_wf.get_probability_density()
        density_state_0 = np.abs(get_ideal_bitflip(initial_wf)) ** 2

        ax.plot(
            grid.x,
            density_state_0,
            color="gray",
            linestyle=":",
            lw=1.5,
            alpha=0.5,
            label=r"Basis State $|0\rangle$",
        )
        ax.plot(
            grid.x,
            density_state_1,
            color="gray",
            linestyle="--",
            lw=1.5,
            alpha=0.5,
            label=r"Basis State $|1\rangle$",
        )

    if T_h is not None:
        density_hadamard = np.abs(get_ideal_hadamard(initial_wf)) ** 2
        ax.plot(
            grid.x,
            density_hadamard,
            color="green",
            linestyle="-.",
            lw=1.5,
            alpha=0.5,
            label=r"Hadamard State $(|0\rangle + |1\rangle) / \sqrt{2}$",
        )

    ax.legend(loc="upper right", framealpha=0.8)

    # Pre-setup the physical state container
    from wavefunction import Wavefunction

    current_wf = Wavefunction(grid)

    # --- 2. Build the Smooth Time Grid ---
    frame_interval_ms = int(1000 / fps)
    pause_frames_count = fps

    # High temporal resolution steps for fluid propagation
    num_active_frames = 500
    dt = total_time / num_active_frames

    time_array = []
    paused_th_multipliers = set()
    paused_tf_multipliers = set()

    t = 0.0
    while t <= total_time + 1e-9:
        time_array.append(t)

        if len(time_array) > 1:
            t_prev = time_array[-2]

            # Check T_h Multiples
            if T_h is not None and T_h > 0:
                prev_mult = int(t_prev // T_h)
                curr_mult = int(t // T_h)
                if curr_mult > prev_mult and curr_mult not in paused_th_multipliers:
                    exact_t = curr_mult * T_h
                    time_array[-1] = exact_t
                    time_array.extend([exact_t] * pause_frames_count)
                    paused_th_multipliers.add(curr_mult)

            # Check T_f Multiples
            if T_f is not None and T_f > 0:
                prev_mult = int(t_prev // T_f)
                curr_mult = int(t // T_f)
                if curr_mult > prev_mult and curr_mult not in paused_tf_multipliers:
                    exact_t = curr_mult * T_f
                    time_array[-1] = exact_t
                    time_array.extend([exact_t] * pause_frames_count)
                    paused_tf_multipliers.add(curr_mult)

        t += dt

    total_extended_frames = len(time_array)

    # --- 3. Animation Framework Core ---
    def init():
        line.set_data(grid.x, np.zeros_like(grid.x))
        time_text.set_text("")
        return line, time_text

    def update(frame):
        t = time_array[frame]

        # Calculate exact quantum evolution step
        prop.evolve(t, current_wf)
        density = current_wf.get_probability_density()

        # Update line data points smoothly
        line.set_ydata(density)

        # Modify string label if the clock is paused at gate milestones
        label_suffix = ""
        if T_f is not None and np.isclose(t % T_f, 0, atol=1e-5) and t > 0:
            multiplier = round(t / T_f)
            label_suffix = f" ({multiplier}$T_f$ Bit Flip Hold)"
        elif T_h is not None and np.isclose(t % T_h, 0, atol=1e-5) and t > 0:
            multiplier = round(t / T_h)
            label_suffix = f" ({multiplier}$T_h$ Hadamard Hold)"

        time_text.set_text(f"Time $t = {t:.2f}${label_suffix}")
        return line, time_text

    # --- 4. Run the Animation Engine ---
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_extended_frames,
        interval=frame_interval_ms,
        blit=True,  # Works efficiently now with fill removed
    )

    plt.show()
    return anim


def plot_waveform_evolution_2d(grid, wf_initial, prop, n_steps, max_time, cmap="magma"):
    """
    Plots a 2D heatmap of the electron's probability density over time,
    displaying the full time evolution continuously without animation holds.

    Args:
        time_array (np.ndarray): 1D array of time steps.
        grid_x (np.ndarray): 1D array of spatial grid points.
        density_matrix (np.ndarray): 2D array of shape (len(time_array), len(grid_x))
                                     containing probability densities |psi(x,t)|^2.
        cmap (str): Colormap to match your desired color scheme.
    """
    # Define how many time steps to calculate, and the time range
    time_array = np.linspace(0, max_time, n_steps)

    # Initialize an empty list to store the time slices
    density_history = []
    temp_wf = Wavefunction(grid)

    # Ensure the propagator is starting from our designated initial state
    prop.set_initial_state(wf_initial)

    for t in time_array:
        # Evolve the state to absolute time t and store in temp_wf
        prop.evolve(t, temp_wf)

        # Extract probability density (fallback to np.abs if method isn't defined)
        if hasattr(temp_wf, "get_probability_density"):
            density = temp_wf.get_probability_density()
        else:
            density = np.abs(temp_wf.psi) ** 2

        density_history.append(density)

    # Convert the list of 1D arrays into a single 2D NumPy array
    density_matrix = np.array(density_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a 2D coordinate grid for the time and position axes
    X, T = np.meshgrid(grid.x, time_array)

    # Plot the density as a 2D continuous heatmap
    # shading='auto' ensures smooth rendering across the grid points
    c = ax.pcolormesh(T, X, density_matrix, cmap=cmap, shading="auto")

    # Formatting and labels
    ax.set_title("Wavefunction Probability Density Evolution", fontsize=14)
    ax.set_ylabel("Position (x)", fontsize=12)
    ax.set_xlabel("Time (t)", fontsize=12)

    # Add a colorbar to represent the density magnitude
    cbar = fig.colorbar(c, ax=ax)

    # Use raw string for LaTeX formatting to avoid syntax warnings
    cbar.set_label(r"$|\psi(x,t)|^2$", fontsize=12)

    plt.tight_layout()
    plt.show()
