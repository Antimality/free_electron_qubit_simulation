import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import qutip as qt


# ==========================================
# Helper Functions for UI & Time
# ==========================================

def _build_time_array(total_time, fps):
    frame_interval_ms = int(1000 / fps)
    num_active_frames = 1000
    time_array = np.linspace(0, total_time, num_active_frames)
    return time_array, frame_interval_ms

# ==========================================
# 2-Qubit Pre-computation
# ==========================================

def _precompute_evolution_2q(grid, prop1, prop2, wf1, wf2, time_array):
    """
    Pre-calculates the wavefunction density and Bloch states for TWO qubits.
    """
    print("Pre-computing quantum evolution for 2 qubits... Please wait.")
    densities1, densities2 = [], []
    bloch_states1, bloch_states2 = [], []
    
    for t in time_array:
        # אבולוציה של קיוביט 1
        prop1.evolve(t, wf1)
        densities1.append(wf1.get_probability_density())
        bloch_states1.append(wf1.get_qutip_qubit())
        
        # אבולוציה של קיוביט 2
        prop2.evolve(t, wf2)
        densities2.append(wf2.get_probability_density())
        bloch_states2.append(wf2.get_qutip_qubit())

    return (densities1, densities2), (bloch_states1, bloch_states2)

# ==========================================
# Setup & Update Functions
# ==========================================

def _setup_1d_axis_2q(ax, grid, y_lim=None):
    """Configures the 1D spatial evolution subplot for TWO qubits."""
    ax.set_xlim(0, grid.L)
    if y_lim is not None:
        ax.set_ylim(0, y_lim)
    ax.set_xlabel("Position (x)", fontsize=12)
    ax.set_ylabel(r"Probability Density $|\psi(x,t)|^2$", fontsize=12)
    ax.set_title("Spatial Wavefunction Evolution", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    ax.axvline(grid.L / 2, color="red", linestyle="--", alpha=0.6, lw=1.5)

    dummy_y = np.zeros_like(grid.x)
    (line1,) = ax.plot(grid.x, dummy_y, lw=2, color="#1f77b4", label="Qubit 1 (Blue)")
    (line2,) = ax.plot(grid.x, dummy_y, lw=2, color="#ff7f0e", label="Qubit 2 (Orange)")

    ax.legend(loc="upper right", framealpha=0.8)
    return line1, line2

def _setup_bloch_axis_2q(fig, ax=None):
    """Configures the 3D Bloch sphere for TWO qubits."""
    b = qt.Bloch(fig=fig, axes=ax)
    
    b.vector_color = ["#1f77b4", "#ff7f0e"] 
    b.point_marker = ["o"]
    b.font_size = 14

    if ax is None:
        b.zlpos = [1.1, -1.2]

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

def _update_bloch_sphere_2q(b, state1, state2):
    """Updates the Bloch sphere with two states simultaneously."""
    b.clear()
    b.add_states([state1, state2]) 
    b.make_sphere()

# ==========================================
# Animation Controller
# ==========================================

def _build_interactive_animation(fig, time_array, update_func, frame_interval_ms, target_t=None):
    fig.subplots_adjust(bottom=0.25)
    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    
    # סידור הכפתורים הבסיסיים
    ax_play   = fig.add_axes([0.20, 0.04, 0.10, 0.05])
    ax_step   = fig.add_axes([0.32, 0.04, 0.10, 0.05])
    ax_reset  = fig.add_axes([0.44, 0.04, 0.10, 0.05])

    slider = Slider(ax_slider, "Time", 0, time_array[-1], valinit=0, color="#003366")
    btn_play = Button(ax_play, "Pause", hovercolor="0.975")
    btn_step = Button(ax_step, "Step", hovercolor="0.975")
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.975")

    btn_reset.on_clicked(lambda event: slider.set_val(0.0))

    # הוספת כפתור קפיצה מותאם אישית אם סופק זמן מטרה
    btn_target = None
    if target_t is not None:
        ax_target = fig.add_axes([0.56, 0.04, 0.15, 0.05])
        btn_target = Button(ax_target, "Hadamard", hovercolor="0.975")
        btn_target.on_clicked(lambda event: slider.set_val(target_t))

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
        fig, wrapped_update, frames=frame_generator, interval=frame_interval_ms, blit=False, save_count=len(time_array),
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

    return anim, (slider, btn_play, btn_step, btn_reset, btn_target, state)   

# ==========================================
# Main Combined Animation
# ==========================================

def animate_combined_simulation_2q(
    grid, prop1, prop2, wf1, wf2, total_time, fps=30, y_lim=None, target_t=None
):
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    line1, line2 = _setup_1d_axis_2q(ax1, grid, y_lim=y_lim)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    b = _setup_bloch_axis_2q(fig, ax=ax2)

    time_text = fig.text(
        0.5, 0.88, "", ha="center", va="center", fontsize=14,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    time_array, frame_interval_ms = _build_time_array(total_time, fps)

    (densities1, densities2), (bloch_states1, bloch_states2) = _precompute_evolution_2q(
        grid, prop1, prop2, wf1, wf2, time_array
    )
    
    state_cache = {"last_t": -1.0}

    def update(frame):
        t = time_array[frame]
        if t == state_cache["last_t"]:
            return []
        state_cache["last_t"] = t

        line1.set_ydata(densities1[frame])
        line2.set_ydata(densities2[frame])
        _update_bloch_sphere_2q(b, bloch_states1[frame], bloch_states2[frame])
        time_text.set_text(f"Time $t = {t:.4f}$")

        return [line1, line2]

    anim, controls = _build_interactive_animation(
        fig, time_array, update, frame_interval_ms, target_t=target_t
    )

    fig.subplots_adjust(bottom=0.20, top=0.85)
    plt.show()

    return anim, controls