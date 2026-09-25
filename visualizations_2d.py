import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


def get_local_qutip_qubit(wf, center_q1, center_q2):
    """Calculates local density matrix"""
    density = np.abs(wf.psi) ** 2
    dx2 = wf.grid.dx**2

    mask_00 = (wf.grid.x1 < center_q1) & (wf.grid.x2 < center_q2)
    mask_01 = (wf.grid.x1 < center_q1) & (wf.grid.x2 >= center_q2)
    mask_10 = (wf.grid.x1 >= center_q1) & (wf.grid.x2 < center_q2)
    mask_11 = (wf.grid.x1 >= center_q1) & (wf.grid.x2 >= center_q2)

    masks = [mask_00, mask_01, mask_10, mask_11]
    probs = [np.sum(density[m]) * dx2 for m in masks]

    amplitudes = []
    for p, mask in zip(probs, masks):
        phase = np.angle(np.sum(wf.psi[mask])) if p > 1e-12 else 0.0
        amplitudes.append(np.sqrt(p) * np.exp(1j * phase))

    return qt.Qobj(np.array(amplitudes).reshape(4, 1), dims=[[2, 2], [1, 1]]).unit()


def _build_interactive_animation(
    fig,
    time_array,
    update_func,
    frame_interval_ms,
    T_target=None,
    jump_btn_text="Jump to Gate",
):
    """Builds the lower user interface (slider, play buttons, jump to time)"""
    fig.subplots_adjust(bottom=0.25)

    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_play = fig.add_axes([0.15, 0.04, 0.10, 0.05])
    ax_step = fig.add_axes([0.27, 0.04, 0.10, 0.05])
    ax_reset = fig.add_axes([0.39, 0.04, 0.10, 0.05])

    slider = Slider(ax_slider, "Time", 0, time_array[-1], valinit=0, color="#003366")
    btn_play = Button(ax_play, "Pause", hovercolor="0.975")
    btn_step = Button(ax_step, "Step", hovercolor="0.975")
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.975")

    btn_reset.on_clicked(lambda event: slider.set_val(0.0))

    btn_jump = None
    if T_target is not None:
        ax_jump = fig.add_axes([0.51, 0.04, 0.18, 0.05])
        btn_jump = Button(ax_jump, jump_btn_text, hovercolor="0.975")
        btn_jump.on_clicked(lambda event: slider.set_val(T_target))

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

    return anim, (slider, btn_play, btn_step, btn_reset, btn_jump, state)


def visual_results_2d(
    wf,
    center_q1,
    center_q2,
    mu,
    sigma,
    D,
    fidelities_tuple,
    target_labels=("|+>", "|0>"),
    title_str="Final State Snapshot",
):
    """
    Creates a snapshot of the final state, with support for dynamic labels for different gates.
    """
    fig = plt.figure(figsize=(15, 6))
    plt.suptitle(title_str, fontsize=16, fontweight="bold")

    ax_spatial = fig.add_subplot(1, 2, 1)
    ax_bloch = fig.add_subplot(1, 2, 2, projection="3d")
    b = qt.Bloch(fig=fig, axes=ax_bloch)

    # Margins for final state
    density_2d = np.abs(wf.psi) ** 2
    marginal_1 = np.sum(density_2d, axis=1) * wf.grid.dx
    marginal_2 = np.sum(density_2d, axis=0) * wf.grid.dx

    ax_spatial.plot(
        wf.grid.x, marginal_1, color="#E63946", lw=2.5, label="Qubit 1 (Target)"
    )
    ax_spatial.plot(
        wf.grid.x,
        marginal_2,
        color="#457B9D",
        lw=2.5,
        linestyle="--",
        label="Qubit 2 (Control)",
    )

    # Spacial division lines
    ax_spatial.hlines(
        0,
        center_q1 - mu - sigma,
        center_q1 - mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
        label="Q1 |0> / |1>",
    )
    ax_spatial.hlines(
        0,
        center_q1 + mu - sigma,
        center_q1 + mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q2 - mu - sigma,
        center_q2 - mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
        label="Q2 |0> / |1>",
    )
    ax_spatial.hlines(
        0,
        center_q2 + mu - sigma,
        center_q2 + mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.axvline(
        center_q1, color="green", linestyle="-.", alpha=0.5, label="Q1 Center"
    )
    ax_spatial.axvline(
        center_q2, color="green", linestyle="-.", alpha=0.5, label="Q2 Center"
    )

    ax_spatial.set_xlim(-D / 2 - 1.2, D / 2 + 1.2)
    ax_spatial.set_ylim(-0.2, max(np.max(marginal_1), np.max(marginal_2)) * 1.3)
    ax_spatial.legend(loc="upper right", fontsize=9)
    ax_spatial.grid(True, alpha=0.2)

    # Fidelity display
    fid_q1, fid_q2 = fidelities_tuple
    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
    ax_spatial.text(
        0.02,
        0.95,
        f"Final Fidelities:\nQ1 (Target {target_labels[0]}): {fid_q1:.4f}\nQ2 (Target {target_labels[1]}): {fid_q2:.4f}",
        transform=ax_spatial.transAxes,
        fontsize=12,
        bbox=props,
    )

    # Bloch Sphere update
    state_2q = get_local_qutip_qubit(wf, center_q1, center_q2)
    rho_2q = state_2q * state_2q.dag()
    rho_1 = rho_2q.ptrace(0)
    rho_2 = rho_2q.ptrace(1)

    vec1 = [
        qt.expect(qt.sigmax(), rho_1),
        qt.expect(qt.sigmay(), rho_1),
        qt.expect(qt.sigmaz(), rho_1),
    ]
    vec2 = [
        qt.expect(qt.sigmax(), rho_2),
        qt.expect(qt.sigmay(), rho_2),
        qt.expect(qt.sigmaz(), rho_2),
    ]

    b.add_vectors(vec1)
    b.add_vectors(vec2)
    b.vector_color = ["#E63946", "#457B9D"]
    b.make_sphere()

    plt.tight_layout()
    plt.show()


def animate_combined_simulation_2d(
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
    sigma,
    D,
    T_hold_start,
    T_gate_duration,
    fps=30,
    gate_name="HADAMARD",
    gate_color="#E63946",
    target_labels=("|+>", "|0>"),
    title_str="Interactive Evolution",
    jump_btn_text="Jump to Gate",
):
    """
    Generic animation supporting variable gate names, colors, and button text.
    """
    fig = plt.figure(figsize=(15, 7.5))
    ax_spatial = fig.add_subplot(1, 2, 1)
    ax_bloch = fig.add_subplot(1, 2, 2, projection="3d")
    b = qt.Bloch(fig=fig, axes=ax_bloch)
    b.make_sphere()

    (line_p1,) = ax_spatial.plot([], [], color="#E63946", lw=2.5, label="Qubit 1")
    (line_p2,) = ax_spatial.plot(
        [], [], color="#457B9D", lw=2.5, linestyle="--", label="Qubit 2"
    )

    ax_spatial.hlines(
        0,
        center_q1 - mu - sigma,
        center_q1 - mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q1 + mu - sigma,
        center_q1 + mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q2 - mu - sigma,
        center_q2 - mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q2 + mu - sigma,
        center_q2 + mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.axvline(center_q1, color="green", linestyle="-.", alpha=0.5)
    ax_spatial.axvline(center_q2, color="green", linestyle="-.", alpha=0.5)

    ax_spatial.set_xlim(-D / 2 - 1.2, D / 2 + 1.2)
    ax_spatial.set_ylim(-0.2, max(np.max(marginals_1), np.max(marginals_2)) * 1.3)
    ax_spatial.set_title(title_str, fontsize=14)
    ax_spatial.legend(loc="upper right", fontsize=9)
    ax_spatial.grid(True, alpha=0.2)

    time_text = ax_spatial.text(
        0.02, 0.95, "", transform=ax_spatial.transAxes, fontsize=11
    )
    gate_text = ax_spatial.text(
        0.02, 0.90, "", transform=ax_spatial.transAxes, fontsize=12, fontweight="bold"
    )
    fid_text = ax_spatial.text(
        0.02,
        0.82,
        "",
        transform=ax_spatial.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    def update(frame):
        t_current = time_array[frame]

        line_p1.set_data(grid2d.x, marginals_1[frame])
        line_p2.set_data(grid2d.x, marginals_2[frame])

        time_text.set_text(f"Time $t = {t_current:.2f}$")
        fid_text.set_text(
            f"Fidelities:\nQ1 ({target_labels[0]}): {fidelities_q1[frame]:.4f}\nQ2 ({target_labels[1]}): {fidelities_q2[frame]:.4f}"
        )

        if t_current < T_hold_start:
            gate_text.set_text("Active Gate: HOLD")
            gate_text.set_color("gray")
        elif t_current < T_hold_start + T_gate_duration:
            gate_text.set_text(f"Active Gate: {gate_name}")
            gate_text.set_color(gate_color)
        else:
            gate_text.set_text("Active Gate: HOLD (Gate Complete)")
            gate_text.set_color("green")

        b.vectors = []
        b.add_vectors(bloch_vectors_1[frame])
        b.add_vectors(bloch_vectors_2[frame])
        b.vector_color = ["#E63946", "#457B9D"]
        b.make_sphere()

        return line_p1, line_p2, time_text, gate_text, fid_text

    frame_interval_ms = int(1000 / fps)
    anim, controls = _build_interactive_animation(
        fig,
        time_array,
        update,
        frame_interval_ms,
        T_target=(T_hold_start + T_gate_duration),
        jump_btn_text=jump_btn_text,
    )
    plt.show()
    return anim, controls


def animate_cnot_circuit_2d(
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
    sigma,
    D,
    time_schedule,
    fps=30,
):
    """
    Advanced animation including 3 panels: dynamic circuit diagram (top),
    physical space (bottom left), and Bloch sphere (bottom right).
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.5])

    # Create subplots
    ax_circuit = fig.add_subplot(gs[0, :])
    ax_spatial = fig.add_subplot(gs[1, 0])
    ax_bloch = fig.add_subplot(gs[1, 1], projection="3d")
    b = qt.Bloch(fig=fig, axes=ax_bloch)
    b.make_sphere()

    # ---------------------------------------------------------
    # Circuit Diagram
    # ---------------------------------------------------------
    ax_circuit.axis("off")
    ax_circuit.set_xlim(-1, 5)
    ax_circuit.set_ylim(-0.5, 1.5)

    # Main lines (Qubits)
    ax_circuit.plot([0, 4.5], [1, 1], color="black", lw=2, zorder=1)  # Q2 (Control)
    ax_circuit.plot([0, 4.5], [0, 0], color="black", lw=2, zorder=1)  # Q1 (Target)
    ax_circuit.text(
        -0.2, 1, "Q2 (Control)", va="center", ha="right", fontsize=12, fontweight="bold"
    )
    ax_circuit.text(
        -0.2, 0, "Q1 (Target)", va="center", ha="right", fontsize=12, fontweight="bold"
    )

    # H1 gate
    rect_h1 = plt.Rectangle(
        (0.7, -0.3),
        0.6,
        0.6,
        fill=True,
        facecolor="white",
        edgecolor="black",
        lw=2.5,
        zorder=3,
    )
    ax_circuit.add_patch(rect_h1)
    text_h1 = ax_circuit.text(
        1.0,
        0,
        "H",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="black",
        zorder=4,
    )

    # CZ gate
    (line_cz,) = ax_circuit.plot([2.5, 2.5], [0, 1], color="black", lw=2.5, zorder=2)
    dot_cz1 = ax_circuit.scatter([2.5], [1.0], color="black", s=150, zorder=3)
    dot_cz2 = ax_circuit.scatter([2.5], [0.0], color="black", s=150, zorder=3)

    # H2 gate
    rect_h2 = plt.Rectangle(
        (3.7, -0.3),
        0.6,
        0.6,
        fill=True,
        facecolor="white",
        edgecolor="black",
        lw=2.5,
        zorder=3,
    )
    ax_circuit.add_patch(rect_h2)
    text_h2 = ax_circuit.text(
        4.0,
        0,
        "H",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="black",
        zorder=4,
    )

    # ---------------------------------------------------------
    # Draw physical space panel
    # ---------------------------------------------------------
    (line_p1,) = ax_spatial.plot([], [], color="#E63946", lw=2.5, label="Q1 (Target)")
    (line_p2,) = ax_spatial.plot(
        [], [], color="#457B9D", lw=2.5, linestyle="--", label="Q2 (Control)"
    )

    ax_spatial.hlines(
        0,
        center_q1 - mu - sigma,
        center_q1 - mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q1 + mu - sigma,
        center_q1 + mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q2 - mu - sigma,
        center_q2 - mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q2 + mu - sigma,
        center_q2 + mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
    )

    ax_spatial.set_xlim(-D / 2 - 1.2, D / 2 + 1.2)
    ax_spatial.set_ylim(-0.2, max(np.max(marginals_1), np.max(marginals_2)) * 1.3)
    ax_spatial.legend(loc="upper right", fontsize=9)
    ax_spatial.grid(True, alpha=0.2)

    time_text = ax_spatial.text(
        0.02, 0.95, "", transform=ax_spatial.transAxes, fontsize=11
    )
    fid_text = ax_spatial.text(
        0.02,
        0.85,
        "",
        transform=ax_spatial.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    # Scheduled time for each gate operation
    t_h1_start, _, t_cz_start, _, t_h2_start, _ = time_schedule

    def update(frame):
        t = time_array[frame]

        # Update Graphs
        line_p1.set_data(grid2d.x, marginals_1[frame])
        line_p2.set_data(grid2d.x, marginals_2[frame])
        time_text.set_text(f"Time $t = {t:.2f}$")
        fid_text.set_text(
            f"Fidelity to Target State (|11>):\nQ1 (|1>): {fidelities_q1[frame]:.4f}\nQ2 (|1>): {fidelities_q2[frame]:.4f}"
        )

        # --- Update gate colors based on time ---
        blue_col = "#457B9D"

        # H1 gate
        if t >= t_h1_start:
            rect_h1.set_edgecolor(blue_col)
            text_h1.set_color(blue_col)
        else:
            rect_h1.set_edgecolor("black")
            text_h1.set_color("black")

        # CZ gate
        if t >= t_cz_start:
            line_cz.set_color(blue_col)
            dot_cz1.set_color(blue_col)
            dot_cz2.set_color(blue_col)
        else:
            line_cz.set_color("black")
            dot_cz1.set_color("black")
            dot_cz2.set_color("black")

        # H2 gate
        if t >= t_h2_start:
            rect_h2.set_edgecolor(blue_col)
            text_h2.set_color(blue_col)
        else:
            rect_h2.set_edgecolor("black")
            text_h2.set_color("black")

        # Update Bloch sphere
        b.vectors = []
        b.add_vectors(bloch_vectors_1[frame])
        b.add_vectors(bloch_vectors_2[frame])
        b.vector_color = ["#E63946", "#457B9D"]
        b.make_sphere()

        return line_p1, line_p2, time_text, fid_text

    frame_interval_ms = int(1000 / fps)
    anim, controls = _build_interactive_animation(
        fig,
        time_array,
        update,
        frame_interval_ms,
        T_target=t_h1_start,
        jump_btn_text="Restart Sequence",
    )
    plt.show()
    return anim, controls


def _build_dynamic_interactive_animation(
    fig,
    time_array,
    update_func,
    frame_interval_ms,
    gate_schedule=None,
):
    """Builds the expanded lower user interface with frame stepping and gate jumping."""
    fig.subplots_adjust(bottom=0.25)

    # UI Axis Definitions
    ax_step_b = fig.add_axes([0.10, 0.12, 0.04, 0.03])
    ax_slider = fig.add_axes([0.15, 0.12, 0.70, 0.03])
    ax_step_f = fig.add_axes([0.86, 0.12, 0.04, 0.03])

    ax_play = fig.add_axes([0.22, 0.04, 0.12, 0.05])
    ax_reset = fig.add_axes([0.37, 0.04, 0.10, 0.05])
    ax_end = fig.add_axes([0.50, 0.04, 0.10, 0.05])
    ax_next = fig.add_axes([0.63, 0.04, 0.15, 0.05])

    # Instantiate Elements
    btn_step_b = Button(ax_step_b, "<", hovercolor="0.975")
    slider = Slider(ax_slider, "", 0, time_array[-1], valinit=0, color="#003366")
    btn_step_f = Button(ax_step_f, ">", hovercolor="0.975")

    btn_play = Button(ax_play, "Pause", hovercolor="0.975")
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.975")
    btn_end = Button(ax_end, "End", hovercolor="0.975")
    btn_next = Button(ax_next, "Next Gate", hovercolor="0.975")

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

    def step_back(event):
        if state.playing:
            toggle_play(None)
        state.frame = (state.frame - 1) % len(time_array)
        wrapped_update(state.frame)
        fig.canvas.draw_idle()

    def step_forward(event):
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

    def reset_sim(event):
        slider.set_val(0.0)

    def jump_end(event):
        slider.set_val(time_array[-1])

    def jump_next_gate(event):
        if state.playing:
            toggle_play(None)
        current_t = time_array[state.frame]
        target_idx = len(time_array) - 1

        if gate_schedule:
            for gate in gate_schedule:
                if gate["end"] > current_t + 0.005:
                    target_idx = (np.abs(time_array - gate["end"])).argmin() + 1
                    break

        state.frame = target_idx
        wrapped_update(target_idx)
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)
    btn_step_b.on_clicked(step_back)
    btn_step_f.on_clicked(step_forward)
    btn_reset.on_clicked(reset_sim)
    btn_end.on_clicked(jump_end)
    btn_next.on_clicked(jump_next_gate)
    slider.on_changed(scrub)

    return anim, (
        slider,
        btn_play,
        btn_step_b,
        btn_step_f,
        btn_reset,
        btn_end,
        btn_next,
        state,
    )


def animate_quantum_circuit_2d(
    grid2d,
    time_array,
    marginals_1,
    marginals_2,
    bloch_vectors_1,
    bloch_vectors_2,
    fid_q1,
    fid_q2,
    center_q1,
    center_q2,
    mu,
    sigma,
    D,
    gate_schedule,
    initial_state,
    target_state,
    fps=30,
):
    """
    General, dynamic circuit animator mapped natively from arbitrary sequence arrays.
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.5])

    ax_circuit = fig.add_subplot(gs[0, :])
    ax_spatial = fig.add_subplot(gs[1, 0])
    ax_bloch = fig.add_subplot(gs[1, 1], projection="3d")
    b = qt.Bloch(fig=fig, axes=ax_bloch)
    b.make_sphere()

    # ---------------------------------------------------------
    # 1. Dynamic Circuit Diagram (Top)
    # ---------------------------------------------------------
    ax_circuit.axis("off")
    ax_circuit.set_ylim(-0.5, 2)

    num_gates = len(gate_schedule)
    x_max = max(num_gates + 1, 3)

    # To cleanly center the circuit regardless of its length, we apply
    # equal symmetric padding to both the left and right rails.
    circuit_length = x_max - 0.5
    padding = 1.5
    ax_circuit.set_xlim(-padding, circuit_length + padding)

    # Base rails: Q1 is top (y=1), Q2 is bottom (y=0)
    ax_circuit.plot([0, circuit_length], [1, 1], color="black", lw=2, zorder=1)
    ax_circuit.plot([0, circuit_length], [0, 0], color="black", lw=2, zorder=1)

    ax_circuit.text(
        -0.2, 1, "Q1", va="center", ha="right", fontsize=14, fontweight="bold"
    )
    ax_circuit.text(
        -0.2, 0, "Q2", va="center", ha="right", fontsize=14, fontweight="bold"
    )

    # Initial State Indicators
    ax_circuit.text(
        -0.8,
        1,
        f"$|{initial_state[0]}\\rangle$",
        va="center",
        ha="right",
        fontsize=15,
        fontweight="bold",
        color="#003366",
    )
    ax_circuit.text(
        -0.8,
        0,
        f"$|{initial_state[1]}\\rangle$",
        va="center",
        ha="right",
        fontsize=15,
        fontweight="bold",
        color="#003366",
    )

    # Final Target State Indicators
    ax_circuit.text(
        circuit_length + 0.8,
        1,
        f"$|{target_state[0]}\\rangle$",
        va="center",
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#2A9D8F",
    )
    ax_circuit.text(
        circuit_length + 0.8,
        0,
        f"$|{target_state[1]}\\rangle$",
        va="center",
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#2A9D8F",
    )

    # Final Fidelities Box at the top of the circuit
    ax_circuit.text(
        circuit_length / 2.0,
        2,
        f"Final Fidelities $\\rightarrow$ Q1 ($|{target_state[0]}\\rangle$): {fid_q1:.4f}   |   Q2 ($|{target_state[1]}\\rangle$): {fid_q2:.4f}",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "wheat",
            "edgecolor": "gray",
            "alpha": 0.8,
        },
    )

    gate_visuals = []

    for i, gate in enumerate(gate_schedule):
        x_pos = i + 1
        name = gate["name"]
        elements = []

        if name == "H1":
            rect = plt.Rectangle(
                (x_pos - 0.3, 0.7),
                0.6,
                0.6,
                fill=True,
                facecolor="white",
                edgecolor="black",
                lw=2.5,
                zorder=3,
            )
            ax_circuit.add_patch(rect)
            txt = ax_circuit.text(
                x_pos,
                1.0,
                "H",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black",
                zorder=4,
            )
            elements.extend([rect, txt])

        elif name == "H2":
            rect = plt.Rectangle(
                (x_pos - 0.3, -0.3),
                0.6,
                0.6,
                fill=True,
                facecolor="white",
                edgecolor="black",
                lw=2.5,
                zorder=3,
            )
            ax_circuit.add_patch(rect)
            txt = ax_circuit.text(
                x_pos,
                0.0,
                "H",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black",
                zorder=4,
            )
            elements.extend([rect, txt])

        elif name == "CZ":
            (line,) = ax_circuit.plot(
                [x_pos, x_pos], [0, 1], color="black", lw=2.5, zorder=2
            )
            (dot1,) = ax_circuit.plot(
                [x_pos], [1.0], marker="o", color="black", markersize=12, zorder=3
            )
            (dot2,) = ax_circuit.plot(
                [x_pos], [0.0], marker="o", color="black", markersize=12, zorder=3
            )
            elements.extend([line, dot1, dot2])

        gate_visuals.append(
            {"start": gate["start"], "end": gate["end"], "elements": elements}
        )

    # ---------------------------------------------------------
    # 2. Spatial Density Panel (Bottom Left)
    # ---------------------------------------------------------
    (line_p1,) = ax_spatial.plot([], [], color="#E63946", lw=2.5, label="Q1")
    (line_p2,) = ax_spatial.plot(
        [], [], color="#457B9D", lw=2.5, linestyle="--", label="Q2"
    )

    # Fixed trap borders
    ax_spatial.hlines(
        0,
        center_q1 - mu - sigma,
        center_q1 - mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q1 + mu - sigma,
        center_q1 + mu + sigma,
        color="#E63946",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q2 - mu - sigma,
        center_q2 - mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
    )
    ax_spatial.hlines(
        0,
        center_q2 + mu - sigma,
        center_q2 + mu + sigma,
        color="#457B9D",
        lw=5,
        alpha=0.4,
    )

    ax_spatial.set_xlim(-D / 2 - 1.2, D / 2 + 1.2)
    ax_spatial.set_ylim(-0.2, max(np.max(marginals_1), np.max(marginals_2)) * 1.3)
    ax_spatial.legend(loc="upper right", fontsize=9)
    ax_spatial.grid(True, alpha=0.2)

    time_text = ax_spatial.text(
        0.02, 0.95, "", transform=ax_spatial.transAxes, fontsize=11
    )

    # ---------------------------------------------------------
    # 3. Dynamic Animation Update Cycle
    # ---------------------------------------------------------
    color_wait = "black"
    color_active = "#457B9D"
    color_done = "#2A9D8F"

    def update(frame):
        t = time_array[frame]

        # Update Spatial Marginals
        line_p1.set_data(grid2d.x, marginals_1[frame])
        line_p2.set_data(grid2d.x, marginals_2[frame])
        time_text.set_text(f"Time $t = {t:.2f}$")

        # Update Circuit Graphic Colors
        for gv in gate_visuals:
            if t < gv["start"]:
                color = color_wait
            elif t <= gv["end"]:
                color = color_active
            else:
                color = color_done

            for el in gv["elements"]:
                if isinstance(el, plt.Rectangle):
                    el.set_edgecolor(color)
                else:
                    el.set_color(color)

        # Update Bloch Sphere Trajectories
        b.vectors = []
        b.add_vectors(bloch_vectors_1[frame])
        b.add_vectors(bloch_vectors_2[frame])
        b.vector_color = ["#E63946", "#457B9D"]
        b.make_sphere()

        return line_p1, line_p2, time_text

    frame_interval_ms = int(1000 / fps)
    anim, controls = _build_dynamic_interactive_animation(
        fig, time_array, update, frame_interval_ms, gate_schedule=gate_schedule
    )

    print(f"Final Fidelity to |{target_state[0]}>: {fid_q1:.4f}")
    print(f"Final Fidelity to |{target_state[1]}>: {fid_q2:.4f}")

    plt.show()
    return anim, controls
