import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from joblib import Memory
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from grid import Grid
from potentials import complex_absorbing_potential
from propagator import SplitOperatorPropagator
from wavefunction import Wavefunction

memory = Memory(location=".sim_cache", verbose=1)

# ==========================================
# Adapted Visualization Functions
# ==========================================


def _build_interactive_animation(fig, time_array, update_func, fps, T_f=None):
    """Wraps the FuncAnimation engine with a playback controller."""
    fig.subplots_adjust(bottom=0.25)
    frame_interval_ms = int(1000 / fps)

    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_play = fig.add_axes([0.15, 0.04, 0.10, 0.05])
    ax_step = fig.add_axes([0.27, 0.04, 0.10, 0.05])
    ax_reset = fig.add_axes([0.39, 0.04, 0.10, 0.05])

    slider = Slider(ax_slider, "Time", 0, time_array[-1], valinit=0, color="#003366")
    btn_play = Button(ax_play, "Pause", hovercolor="0.975")
    btn_step = Button(ax_step, "Step", hovercolor="0.975")
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.975")

    btn_reset.on_clicked(lambda event: slider.set_val(0.0))

    btn_f = None
    if T_f is not None and T_f > 0:
        ax_f = fig.add_axes([0.51, 0.04, 0.15, 0.05])
        btn_f = Button(ax_f, "Bit Flip", hovercolor="0.975")
        btn_f.on_clicked(lambda event: slider.set_val(T_f))

    class State:
        frame = 0
        playing = True
        ignore_slider = False

    state = State()

    def frame_generator():
        while True:
            if state.playing:
                state.frame = (state.frame + 1) % len(time_array)
            yield state.frame

    def wrapped_update(frame_idx):
        artists = update_func(frame_idx)
        if not state.ignore_slider:
            state.ignore_slider = True
            slider.set_val(time_array[frame_idx])
            state.ignore_slider = False
        return artists if artists else []

    anim = FuncAnimation(
        fig,
        wrapped_update,
        frames=frame_generator,
        interval=frame_interval_ms,
        blit=False,
        save_count=len(time_array),
    )

    def toggle_play(event):
        if state.playing:
            anim.pause()
            btn_play.label.set_text("Play")
        else:
            anim.resume()
            btn_play.label.set_text("Pause")
        state.playing = not state.playing
        fig.canvas.draw_idle()

    def step_frame(event):
        if state.playing:
            toggle_play(None)
        state.frame = (state.frame + 1) % len(time_array)
        wrapped_update(state.frame)
        fig.canvas.draw_idle()

    def scrub(val):
        if state.ignore_slider:
            return
        if state.playing:
            toggle_play(None)
        idx = (np.abs(time_array - val)).argmin()
        state.frame = idx
        wrapped_update(idx)
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)
    btn_step.on_clicked(step_frame)
    slider.on_changed(scrub)

    return anim, (slider, btn_play, btn_step, btn_reset, btn_f, state)


def visual_results_dynamic(grid, initial_density, final_density, T_f):
    """Statically plots the initial and final states of the wavepacket."""
    _fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plt.suptitle("Passage Through Ponderomotive Focal Plane", fontsize=16)

    # Initial State
    axs[0].plot(grid.x, initial_density, color="#003366", lw=2)
    axs[0].fill_between(grid.x, initial_density, color="#003366", alpha=0.2)
    axs[0].axvline(grid.grid_center, color="red", linestyle="--", alpha=0.6)
    axs[0].set_title("Initial State $|1\\rangle$\n($t=0$)", fontsize=14)
    axs[0].set_ylabel("Probability Density $|\\psi(x)|^2$")
    axs[0].set_xlabel("Position (x)")
    axs[0].set_xlim(grid.x_min, grid.x_max)
    axs[0].grid(alpha=0.3)

    # Final State
    axs[1].plot(grid.x, final_density, color="#003366", lw=2)
    axs[1].fill_between(grid.x, final_density, color="#003366", alpha=0.2)
    axs[1].axvline(grid.grid_center, color="red", linestyle="--", alpha=0.6)
    axs[1].set_title(f"Flipped State $|0\\rangle$\n($t={T_f:.4f}$)", fontsize=14)
    axs[1].set_xlabel("Position (x)")
    axs[1].set_xlim(grid.x_min, grid.x_max)
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_waveform_evolution_2d_dynamic(grid, time_array, density_history, cmap="magma"):
    """Plots the continuous 2D heatmap of the wavepacket evolution."""
    density_matrix = np.array(density_history)
    fig, ax = plt.subplots(figsize=(10, 6))

    X, T = np.meshgrid(grid.x, time_array)
    c = ax.pcolormesh(T, X, density_matrix, cmap=cmap, shading="auto")

    ax.set_title("Wavefunction Probability Density Evolution", fontsize=14)
    ax.set_ylabel("Position (x)", fontsize=12)
    ax.set_xlabel("Time (t)", fontsize=12)

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Probability Density $|\\psi(x,t)|^2$", fontsize=12)

    # Mark the exact focal plane
    focal_time = time_array[-1] / 2.0
    ax.axvline(
        focal_time, color="white", linestyle="--", alpha=0.5, label="SLM Focal Plane"
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def animate_combined_simulation_dynamic(
    grid, time_array, densities, potentials, bloch_states, T_f, fps=30
):
    """Animates the spatial wavefunction, dynamic potential, and Bloch sphere natively."""
    fig = plt.figure(figsize=(16, 8))

    # --- Setup 1D Spatial Axis ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(grid.x_min, grid.x_max)
    ax1.set_ylim(0, 1.5)
    ax1.set_xlabel("Position (x)", fontsize=12)
    ax1.set_ylabel("Probability Density $|\\psi(x,t)|^2$", fontsize=12)
    ax1.set_title("Dynamic Spatial Evolution", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.axvline(grid.grid_center, color="red", linestyle="--", alpha=0.6, lw=1.5)

    (line,) = ax1.plot(
        grid.x, np.zeros_like(grid.x), lw=2, color="#003366", label="Wavepacket"
    )
    (pot_line,) = ax1.plot(
        grid.x,
        np.zeros_like(grid.x),
        color="black",
        linestyle="-",
        lw=2,
        alpha=0.5,
        label="Ponderomotive Lens",
    )
    ax1.legend(loc="upper right")

    # --- Setup 3D Bloch Sphere ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    b = qt.Bloch(fig=fig, axes=ax2)
    b.point_color = ["#ff0000"]
    b.point_marker = ["o"]
    b.point_size = [20]
    b.font_size = 14
    try:
        ax2.disable_mouse_rotation()
    except AttributeError:
        pass

    fig.text(
        0.5,
        0.96,
        f"Focal Plane / Bit Flip expected at $t = {T_f:.4f}$",
        ha="center",
        va="center",
        fontsize=12,
        color="#555555",
    )
    time_text = fig.text(
        0.5,
        0.88,
        "",
        ha="center",
        va="center",
        fontsize=14,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    state_cache = {"last_t": -1.0}

    def update(frame):
        t = time_array[frame]
        if t == state_cache["last_t"]:
            return []
        state_cache["last_t"] = t

        # Update density and dynamic potential
        line.set_ydata(densities[frame])

        # Scale potential down visually so it fits on the same plot as the density
        scaled_pot = np.real(potentials[frame]) * 0.02
        pot_line.set_ydata(scaled_pot)

        # Update Bloch Sphere
        b.clear()
        b.add_states(bloch_states[frame])
        b.make_sphere()

        time_text.set_text(f"Time $t = {t:.4f}$")
        return [line, pot_line]

    anim, controls = _build_interactive_animation(fig, time_array, update, fps, T_f=T_f)
    fig.subplots_adjust(bottom=0.20, top=0.85)
    plt.show()
    return anim, controls


# ==========================================
# Main Dynamic Simulation Loop
# ==========================================


@memory.cache
def run_dynamic_focus_simulation():
    # Constants
    L_grid = 1.0
    N_grid = 1000
    m = 1.0
    hbar = 1.0
    offset = L_grid / 4.0

    # Dynamic Lens Parameters
    omega_peak = 10.0
    t_rayleigh = 0.1
    t_focus = 0.5

    total_time = 2.0 * t_focus
    dt = 0.002
    time_steps = int(total_time / dt)
    time_array = np.linspace(0, total_time, time_steps)

    grid = Grid(L_grid, N_grid, padding_mult=7.0)
    cap = complex_absorbing_potential(grid, width=0.5, strength=100.0)

    wf_current = Wavefunction(grid)
    wf_current.init_coherent_state(offset=offset, omega=omega_peak, m=m, hbar=hbar)

    # Store history for visualization
    densities = []
    potentials = []
    bloch_states = []

    initial_density = np.abs(wf_current.psi) ** 2

    print("Initializing time-dependent Split-Operator Propagator...")
    prop = SplitOperatorPropagator(grid, np.zeros(N_grid), m=m, hbar=hbar)
    prop.set_initial_state(wf_current)

    print("\nSimulating passage through the SLM focal plane...")
    for t in time_array:
        # 1. Calculate dynamic focus constraints
        W_t = np.sqrt(1.0 + ((t - t_focus) / t_rayleigh) ** 2)
        omega_t = omega_peak / (W_t**2)

        # 2. Update potential
        V_real = 0.5 * m * (omega_t**2) * grid.x**2
        V_total = V_real - 1j * cap
        prop.potential = V_total.astype(complex)

        # 3. Step forward and record data
        prop.evolve(t, wf_current, dt=dt)

        densities.append(np.abs(wf_current.psi) ** 2)
        potentials.append(V_real)

        # Use Wavefunction's built-in method to map spatial quadrants to |0> and |1>
        bloch_states.append(wf_current.get_qutip_qubit())

    final_density = np.abs(wf_current.psi) ** 2
    print("Passage complete. Launching visualizers...")

    # 1. Static Initial vs Final State
    visual_results_dynamic(grid, initial_density, final_density, T_f=t_focus)

    # 2. Continuous 2D Heatmap
    plot_waveform_evolution_2d_dynamic(grid, time_array, densities)

    # 3. Interactive UI (Density + Potential + Bloch Sphere)
    animate_combined_simulation_dynamic(
        grid, time_array, densities, potentials, bloch_states, T_f=t_focus, fps=30
    )


if __name__ == "__main__":
    run_dynamic_focus_simulation()
