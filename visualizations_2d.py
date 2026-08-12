import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import qutip as qt

def get_local_qutip_qubit(wf, center_q1, center_q2):
    """מחשב את מטריצות הצפיפות המקומיות (הועבר לכאן לשימוש חוזר)"""
    density = np.abs(wf.psi)**2
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

def _build_interactive_animation(fig, time_array, update_func, frame_interval_ms, T_h_target=None):
    """בונה את ממשק המשתמש התחתון (סליידר, כפתורי נגינה, קפיצה לזמן)"""
    fig.subplots_adjust(bottom=0.25)

    ax_slider = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_play   = fig.add_axes([0.15, 0.04, 0.10, 0.05])
    ax_step   = fig.add_axes([0.27, 0.04, 0.10, 0.05])
    ax_reset  = fig.add_axes([0.39, 0.04, 0.10, 0.05])

    slider = Slider(ax_slider, "Time", 0, time_array[-1], valinit=0, color="#003366")
    btn_play = Button(ax_play, "Pause", hovercolor="0.975")
    btn_step = Button(ax_step, "Step", hovercolor="0.975")
    btn_reset = Button(ax_reset, "Reset", hovercolor="0.975")

    btn_reset.on_clicked(lambda event: slider.set_val(0.0))

    btn_h = None
    if T_h_target is not None:
        ax_h = fig.add_axes([0.51, 0.04, 0.18, 0.05])
        btn_h = Button(ax_h, "Jump to Hadamard", hovercolor="0.975")
        btn_h.on_clicked(lambda event: slider.set_val(T_h_target))

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

    return anim, (slider, btn_play, btn_step, btn_reset, btn_h, state)


def visual_results_2d(wf, center_q1, center_q2, mu, sigma, D, fidelities_tuple):
    """
    מייצר Snapshot סטטי של סוף התהליך (פונקציות שוליות + בלוך + פידליטי).
    """
    fig = plt.figure(figsize=(15, 6))
    plt.suptitle("Final State Snapshot & Fidelities", fontsize=16, fontweight="bold")
    
    ax_spatial = fig.add_subplot(1, 2, 1)
    ax_bloch = fig.add_subplot(1, 2, 2, projection='3d')
    b = qt.Bloch(fig=fig, axes=ax_bloch)
    
    # חישוב השוליות למצב הסופי
    density_2d = np.abs(wf.psi)**2
    marginal_1 = np.sum(density_2d, axis=1) * wf.grid.dx
    marginal_2 = np.sum(density_2d, axis=0) * wf.grid.dx
    
    ax_spatial.plot(wf.grid.x, marginal_1, color="#E63946", lw=2.5, label="Qubit 1 (Target)")
    ax_spatial.plot(wf.grid.x, marginal_2, color="#457B9D", lw=2.5, linestyle="--", label="Qubit 2 (Frozen)")
    
    # סימון אזורים (Underlines)
    ax_spatial.hlines(0, center_q1 - mu - sigma, center_q1 - mu + sigma, color="#E63946", lw=5, alpha=0.4, label="Q1 |0> / |1>")
    ax_spatial.hlines(0, center_q1 + mu - sigma, center_q1 + mu + sigma, color="#E63946", lw=5, alpha=0.4)
    ax_spatial.hlines(0, center_q2 - mu - sigma, center_q2 - mu + sigma, color="#457B9D", lw=5, alpha=0.4, label="Q2 |0> / |1>")
    ax_spatial.hlines(0, center_q2 + mu - sigma, center_q2 + mu + sigma, color="#457B9D", lw=5, alpha=0.4)
    ax_spatial.axvline(center_q1, color="green", linestyle="-.", alpha=0.5, label="Center (|+> state)")
    
    ax_spatial.set_xlim(-D/2 - 1.2, D/2 + 1.2)
    ax_spatial.set_ylim(-0.2, max(np.max(marginal_1), np.max(marginal_2)) * 1.3)
    ax_spatial.legend(loc="upper right", fontsize=9)
    ax_spatial.grid(True, alpha=0.2)
    
    # הצגת פידליטי
    fid_q1, fid_q2 = fidelities_tuple
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax_spatial.text(0.02, 0.95, f"Final Fidelities:\nQ1 (Target |+>): {fid_q1:.4f}\nQ2 (Target |0>): {fid_q2:.4f}", 
                    transform=ax_spatial.transAxes, fontsize=12, bbox=props)

    # עדכון ספירת בלוך
    state_2q = get_local_qutip_qubit(wf, center_q1, center_q2)
    rho_2q = state_2q * state_2q.dag()
    rho_1 = rho_2q.ptrace(0) 
    rho_2 = rho_2q.ptrace(1)
    
    vec1 = [qt.expect(qt.sigmax(), rho_1), qt.expect(qt.sigmay(), rho_1), qt.expect(qt.sigmaz(), rho_1)]
    vec2 = [qt.expect(qt.sigmax(), rho_2), qt.expect(qt.sigmay(), rho_2), qt.expect(qt.sigmaz(), rho_2)]
    
    b.add_vectors(vec1)
    b.add_vectors(vec2)
    b.vector_color = ["#E63946", "#457B9D"]
    b.make_sphere()
    
    plt.tight_layout()
    plt.show()

def animate_combined_simulation_2d(
    grid2d, time_array, marginals_1, marginals_2, 
    bloch_vectors_1, bloch_vectors_2, fidelities_q1, fidelities_q2,
    center_q1, center_q2, mu, sigma, D, 
    T_hold_start, T_h_duration, fps=30
):
    """
    בונה ומריץ את האנימציה האינטראקטיבית מתוך מערכי הנתונים שחושבו מראש.
    """
    fig = plt.figure(figsize=(15, 7.5))
    ax_spatial = fig.add_subplot(1, 2, 1)
    ax_bloch = fig.add_subplot(1, 2, 2, projection='3d')
    b = qt.Bloch(fig=fig, axes=ax_bloch)
    b.make_sphere()
    
    line_p1, = ax_spatial.plot([], [], color="#E63946", lw=2.5, label="Qubit 1")
    line_p2, = ax_spatial.plot([], [], color="#457B9D", lw=2.5, linestyle="--", label="Qubit 2")
    
    ax_spatial.hlines(0, center_q1 - mu - sigma, center_q1 - mu + sigma, color="#E63946", lw=5, alpha=0.4)
    ax_spatial.hlines(0, center_q1 + mu - sigma, center_q1 + mu + sigma, color="#E63946", lw=5, alpha=0.4)
    ax_spatial.hlines(0, center_q2 - mu - sigma, center_q2 - mu + sigma, color="#457B9D", lw=5, alpha=0.4)
    ax_spatial.hlines(0, center_q2 + mu - sigma, center_q2 + mu + sigma, color="#457B9D", lw=5, alpha=0.4)
    ax_spatial.axvline(center_q1, color="green", linestyle="-.", alpha=0.5)
    ax_spatial.axvline(center_q2, color="green", linestyle="-.", alpha=0.5)
    
    ax_spatial.set_xlim(-D/2 - 1.2, D/2 + 1.2)
    ax_spatial.set_ylim(-0.2, max(np.max(marginals_1), np.max(marginals_2)) * 1.3)
    ax_spatial.set_title("Interactive Hadamard Evolution", fontsize=14)
    ax_spatial.legend(loc="upper right", fontsize=9)
    ax_spatial.grid(True, alpha=0.2)
    
    time_text = ax_spatial.text(0.02, 0.95, "", transform=ax_spatial.transAxes, fontsize=11)
    gate_text = ax_spatial.text(0.02, 0.90, "", transform=ax_spatial.transAxes, fontsize=12, fontweight="bold")
    fid_text = ax_spatial.text(0.02, 0.82, "", transform=ax_spatial.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

    def update(frame):
        t_current = time_array[frame]
        
        line_p1.set_data(grid2d.x, marginals_1[frame])
        line_p2.set_data(grid2d.x, marginals_2[frame])
        
        time_text.set_text(f"Time $t = {t_current:.2f}$")
        fid_text.set_text(f"Fidelities:\nQ1 (|+>): {fidelities_q1[frame]:.4f}\nQ2 (|0>): {fidelities_q2[frame]:.4f}")
        
        if t_current < T_hold_start:
            gate_text.set_text("Active Gate: HOLD |00>")
            gate_text.set_color("gray")
        elif t_current < T_hold_start + T_h_duration:
            gate_text.set_text("Active Gate: HADAMARD (Q1)")
            gate_text.set_color("#E63946")
        else:
            gate_text.set_text("Active Gate: HOLD (Superposition)")
            gate_text.set_color("green")
        
        b.vectors = []
        b.add_vectors(bloch_vectors_1[frame])
        b.add_vectors(bloch_vectors_2[frame])
        b.vector_color = ["#E63946", "#457B9D"]
        b.make_sphere()
        
        return line_p1, line_p2, time_text, gate_text, fid_text

    frame_interval_ms = int(1000 / fps)
    anim, controls = _build_interactive_animation(
        fig, time_array, update, frame_interval_ms, T_h_target=(T_hold_start + T_h_duration)
    )
    plt.show()
    return anim, controls