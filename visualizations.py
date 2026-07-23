import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import qutip as qt

from wavefunction import Wavefunction
from metrics import get_fidelity, get_ideal_bitflip, get_ideal_hadamard
from gate_finder import find_bit_flip_time, find_hadamard_time


# ==========================================
# Helper Functions for Code Reuse
# ==========================================


def _build_time_array(total_time, fps):
    """Generates a smooth linear time grid."""
    frame_interval_ms = int(1000 / fps)
    num_active_frames = 1000
    time_array = np.linspace(0, total_time, num_active_frames)
    return time_array, frame_interval_ms


def _add_subtitle(fig, T_h, T_f):
    """Adds a subtitle marking theoretical gate times if they exist."""
    texts = []
    if T_h is not None and T_h > 0:
        texts.append(f"Hadamard expected at $t = {T_h:.4f}$")
    if T_f is not None and T_f > 0:
        texts.append(f"Bit Flip expected at $t = {T_f:.4f}$")

    if texts:
        fig.text(
            0.5,
            0.96,
            " | ".join(texts),
            ha="center",
            va="center",
            fontsize=12,
            color="#555555",
        )


def _precompute_evolution(grid, prop, time_array):
    """
    Pre-calculates the wavefunction density and Bloch states.
    Decouples the heavy physics math from the UI rendering loop.
    """
    print("Pre-computing quantum evolution for smooth playback... Please wait.")
    densities = []
    bloch_states = []
    current_wf = Wavefunction(grid)

    for t in time_array:
        prop.evolve(t, current_wf)
        densities.append(current_wf.get_probability_density())
        bloch_states.append(current_wf.get_qutip_qubit())

    return densities, bloch_states


def _build_interactive_animation(
    fig, time_array, update_func, frame_interval_ms, T_h=None, T_f=None
):
    """
    Wraps the FuncAnimation engine with a playback controller (Slider, Play/Pause, Step, Jump).
    Returns the animation object and a tuple of widgets (to prevent garbage collection).
    """
    # Adjust layout to make room for the UI space
    fig.subplots_adjust(bottom=0.25)

    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_play = fig.add_axes([0.15, 0.04, 0.1, 0.05])
    ax_step = fig.add_axes([0.27, 0.04, 0.1, 0.05])

    slider = Slider(ax_slider, "Time", 0, time_array[-1], valinit=0, color="#003366")
    btn_play = Button(ax_play, "Pause", hovercolor="0.975")
    btn_step = Button(ax_step, "Step", hovercolor="0.975")

    # --- Setup Conditional Jump Buttons ---
    btn_h, btn_f = None, None
    if T_h is not None and T_h > 0:
        ax_h = fig.add_axes([0.39, 0.04, 0.15, 0.05])
        btn_h = Button(ax_h, "Hadamard", hovercolor="0.975")
        btn_h.on_clicked(lambda event: slider.set_val(T_h))

    if T_f is not None and T_f > 0:
        x_pos = 0.56 if (T_h is not None and T_h > 0) else 0.39
        ax_f = fig.add_axes([x_pos, 0.04, 0.15, 0.05])
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

    return anim, (slider, btn_play, btn_step, btn_h, btn_f, state)


def _setup_1d_axis(ax, grid, prop, initial_wf, y_lim=None, T_h=None, T_f=None):
    """Configures the 1D spatial evolution subplot and draws static overlays."""
    ax.set_xlim(0, grid.L)
    if y_lim is not None:
        ax.set_ylim(0, y_lim)
    ax.set_xlabel("Position (x)", fontsize=12)
    ax.set_ylabel(r"Probability Density $|\psi(x,t)|^2$", fontsize=12)
    ax.set_title("Spatial Wavefunction Evolution", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    ax.axvline(grid.L / 2, color="red", linestyle="--", alpha=0.6, lw=1.5)

    if hasattr(prop, "potential"):
        ax.plot(
            grid.x,
            prop.potential,
            color="black",
            linestyle="-",
            lw=2,
            alpha=0.7,
            label="Active Potential",
        )

    dummy_y = np.zeros_like(grid.x)
    (line,) = ax.plot(grid.x, dummy_y, lw=2, color="#003366")

    if T_f is not None:
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
            label=r"Hadamard State",
        )

    ax.legend(loc="upper right", framealpha=0.8)
    return line


def _setup_bloch_axis(fig, ax=None):
    """
    Configures the 3D Bloch sphere subplot without the trajectory history,
    and disables interactive 3D rotation.
    """
    b = qt.Bloch(fig=fig, axes=ax)
    b.point_color = ["#ff0000"]
    b.point_marker = ["o"]
    b.point_size = [20]
    b.font_size = 14

    if ax is None:
        b.zlpos = [1.1, -1.2]

    # Robustly disable mouse rotation
    if b.axes is not None:
        try:
            if isinstance(b.axes, (list, tuple)):
                for a in b.axes:
                    a.disable_mouse_rotation()
            else:
                b.axes.disable_mouse_rotation()
        except AttributeError:
            pass

    return b


def _update_bloch_sphere(b, state):
    """
    Helper function to update the Bloch sphere state natively.
    """
    b.clear()
    b.add_states(state)
    b.make_sphere()


# ==========================================
# Main Visualizations
# ==========================================


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
    assert (wf_h is not None) or (wf_f is not None)

    if wf_h is not None:
        target_h = get_ideal_hadamard(wf_initial)
        fidelity_h = get_fidelity(wf_h.psi, target_h, grid.dx)

    if wf_f is not None:
        target_f = get_ideal_bitflip(wf_initial)
        fidelity_f = get_fidelity(wf_f.psi, target_f, grid.dx)

    if T_h is None:
        T_h, _ = find_hadamard_time(prop, wf_initial, max_time_search=max_search_time)

    if T_f is None:
        T_f, _ = find_bit_flip_time(prop, wf_initial, max_time_search=max_search_time)

    subplots = 1 + (wf_h is not None) + (wf_f is not None)
    fig, axs = plt.subplots(1, subplots, figsize=(18, 5), sharey=True)

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    plt.suptitle("Passive Time Evolution from $|1\\rangle$", fontsize=16)

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
                f"Hadamard State\n($t={T_h:.4f}$)",
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
                f"Flipped state $|0\\rangle$\n($t={T_f:.4f}$)",
                axs[ax_idx],
                fidelity_f,
            )
        )

    for psi, psi_target, title, ax, fidelity in plot_data:
        density_target = np.abs(psi_target) ** 2
        ax.plot(grid.x, density_target, color="green", linestyle=":", alpha=0.8)

        density = np.abs(psi) ** 2
        ax.plot(grid.x, density, color="#003366", lw=2)
        ax.fill_between(grid.x, density, color="#003366", alpha=0.2)
        ax.axvline(grid.L / 2, color="red", linestyle="--", alpha=0.6)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Position (x)")
        ax.set_xlim(0, grid.L)
        ax.grid(alpha=0.3)

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

    axs[0].set_ylabel(r"Probability Density $|\psi(x)|^2$")
    plt.tight_layout()
    plt.show()


def plot_waveform_evolution_2d(grid, wf_initial, prop, n_steps, max_time, cmap="magma"):
    time_array = np.linspace(0, max_time, n_steps)
    density_history = []
    temp_wf = Wavefunction(grid)
    prop.set_initial_state(wf_initial)

    for t in time_array:
        prop.evolve(t, temp_wf)
        if hasattr(temp_wf, "get_probability_density"):
            density = temp_wf.get_probability_density()
        else:
            density = np.abs(temp_wf.psi) ** 2
        density_history.append(density)

    density_matrix = np.array(density_history)
    fig, ax = plt.subplots(figsize=(10, 6))
    X, T = np.meshgrid(grid.x, time_array)
    c = ax.pcolormesh(T, X, density_matrix, cmap=cmap, shading="auto")

    ax.set_title("Wavefunction Probability Density Evolution", fontsize=14)
    ax.set_ylabel("Position (x)", fontsize=12)
    ax.set_xlabel("Time (t)", fontsize=12)

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(r"$|\psi(x,t)|^2$", fontsize=12)

    plt.tight_layout()
    plt.show()


def animate_simulation(
    grid, prop, initial_wf, total_time, T_h=None, T_f=None, fps=30, y_lim=None
):
    fig, ax = plt.subplots(figsize=(10, 7))
    line = _setup_1d_axis(ax, grid, prop, initial_wf, y_lim, T_h, T_f)
    _add_subtitle(fig, T_h, T_f)

    time_array, frame_interval_ms = _build_time_array(total_time, fps)

    # Pre-compute
    densities, _ = _precompute_evolution(grid, prop, time_array)
    state_cache = {"last_t": -1.0}

    def update(frame):
        t = time_array[frame]
        if t == state_cache["last_t"]:
            return []
        state_cache["last_t"] = t

        line.set_ydata(densities[frame])
        return [line]

    anim, controls = _build_interactive_animation(
        fig, time_array, update, frame_interval_ms, T_h=T_h, T_f=T_f
    )
    plt.show()
    return anim, controls


def animate_bloch_sphere(
    grid, prop, initial_wf, total_time, T_h=None, T_f=None, fps=30
):
    fig = plt.figure(figsize=(8, 9))
    ax = fig.add_axes([0.1, 0.25, 0.8, 0.65], projection="3d")
    ax.disable_mouse_rotation()

    b = _setup_bloch_axis(fig, ax=ax)
    _add_subtitle(fig, T_h, T_f)

    time_array, frame_interval_ms = _build_time_array(total_time, fps)

    # Pre-compute
    _, bloch_states = _precompute_evolution(grid, prop, time_array)
    state_cache = {"last_t": -1.0}

    def update(frame):
        t = time_array[frame]
        if t == state_cache["last_t"]:
            return []
        state_cache["last_t"] = t

        _update_bloch_sphere(b, bloch_states[frame])

        return []

    anim, controls = _build_interactive_animation(
        fig, time_array, update, frame_interval_ms, T_h=T_h, T_f=T_f
    )
    plt.show()
    return anim, controls


def animate_combined_simulation(
    grid, prop, initial_wf, total_time, T_h=None, T_f=None, fps=30, y_lim=None
):
    fig = plt.figure(figsize=(16, 8))

    # Left Subplot
    ax1 = fig.add_subplot(1, 2, 1)
    line = _setup_1d_axis(ax1, grid, prop, initial_wf, T_h=T_h, T_f=T_f, y_lim=y_lim)

    # Right Subplot
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    b = _setup_bloch_axis(fig, ax=ax2)

    _add_subtitle(fig, T_h, T_f)

    time_text = fig.text(
        0.5,
        0.88,
        "",
        ha="center",
        va="center",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    time_array, frame_interval_ms = _build_time_array(total_time, fps)

    # Pre-compute
    densities, bloch_states = _precompute_evolution(grid, prop, time_array)
    state_cache = {"last_t": -1.0}

    def update(frame):
        t = time_array[frame]
        if t == state_cache["last_t"]:
            return []
        state_cache["last_t"] = t

        line.set_ydata(densities[frame])
        _update_bloch_sphere(b, bloch_states[frame])
        time_text.set_text(f"Time $t = {t:.4f}$")

        return [line]

    anim, controls = _build_interactive_animation(
        fig, time_array, update, frame_interval_ms, T_h=T_h, T_f=T_f
    )

    fig.subplots_adjust(bottom=0.20, top=0.85)
    plt.show()

    return anim, controls
