import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import qutip as qt

def animate_2qubit_simulation(grid2d, prop2d, wf_initial, total_time, dt_anim, mu, fps=30):
    """
    Animates a 2-electron system.
    Left: 1D projection (marginals) of both electrons with logical state markers.
    Right: Bloch sphere showing the reduced density matrix vectors for each electron.
    """
    print("Pre-computing 2D quantum evolution... Please wait.")
    
    time_steps = int(total_time / dt_anim)
    time_array = np.linspace(0, total_time, time_steps)
    
    # אתחול המנוע
    prop2d.set_initial_state(wf_initial)
    current_wf = type(wf_initial)(grid2d)
    
    # אגירת נתונים מראש כדי שהאנימציה תרוץ חלק
    marginals_1 = []
    marginals_2 = []
    bloch_vectors_1 = []
    bloch_vectors_2 = []
    
    for t in time_array:
        prop2d.evolve(t, current_wf, dt=0.005)
        
        # 1. חישוב ההסתברויות השוליות (Marginals)
        density_2d = current_wf.get_probability_density()
        # סכימה על ציר 1 (x2) כדי לקבל את הסיכוי של x1
        prob1 = np.sum(density_2d, axis=1) * grid2d.dx 
        # סכימה על ציר 0 (x1) כדי לקבל את הסיכוי של x2
        prob2 = np.sum(density_2d, axis=0) * grid2d.dx
        
        marginals_1.append(prob1)
        marginals_2.append(prob2)
        
        # 2. חישוב מצבי הבלוך דרך עקבה חלקית (Partial Trace)
        state_2q = current_wf.get_qutip_qubit() # מחזיר טנזור של שני קיוביטים
        rho_2q = state_2q * state_2q.dag()      # מטריצת הצפיפות המלאה
        
        # עקבה חלקית לחילוץ המצב המצומצם של כל קיוביט
        rho_1 = rho_2q.ptrace(0) 
        rho_2 = rho_2q.ptrace(1)
        
        # חילוץ רכיבי הבלוך (תוחלת של מטריצות פאולי)
        vec1 = [qt.expect(qt.sigmax(), rho_1), qt.expect(qt.sigmay(), rho_1), qt.expect(qt.sigmaz(), rho_1)]
        vec2 = [qt.expect(qt.sigmax(), rho_2), qt.expect(qt.sigmay(), rho_2), qt.expect(qt.sigmaz(), rho_2)]
        
        bloch_vectors_1.append(vec1)
        bloch_vectors_2.append(vec2)

    # ==========================================
    # הגדרת הויזואליזציה (Matplotlib + QuTiP)
    # ==========================================
    fig = plt.figure(figsize=(14, 6))
    
    # צד שמאל: הגריד וההסתברויות (Marginals)
    ax_spatial = fig.add_subplot(1, 2, 1)
    
    # צד ימין: ספירת בלוך
    ax_bloch = fig.add_subplot(1, 2, 2, projection='3d')
    b = qt.Bloch(fig=fig, axes=ax_bloch)
    b.make_sphere()
    
    # קווי השרטוט על הגריד המרחבי
    line_p1, = ax_spatial.plot([], [], color="#E63946", lw=2, label="Electron 1 (Red)")
    line_p2, = ax_spatial.plot([], [], color="#457B9D", lw=2, linestyle="--", label="Electron 2 (Blue)")
    
    # סימון מצבי היסוד (Underlines/Markers)
    ax_spatial.axvline(-mu, color="gray", linestyle=":", alpha=0.7, label="State |0⟩")
    ax_spatial.axvline(mu, color="gray", linestyle=":", alpha=0.7, label="State |1⟩")
    ax_spatial.axvline(0.0, color="green", linestyle="-.", alpha=0.5, label="State |+⟩ (Hadamard)")
    
    # עיצוב הגרף השמאלי
    ax_spatial.set_xlim(grid2d.x[0] / 2, grid2d.x[-1] / 2) # זום-אין לחלק האמצעי
    ax_spatial.set_ylim(0, max(np.max(marginals_1), np.max(marginals_2)) * 1.1)
    ax_spatial.set_title("Marginal Spatial Wavefunctions", fontsize=14)
    ax_spatial.set_xlabel("Position", fontsize=12)
    ax_spatial.set_ylabel("Probability Density", fontsize=12)
    ax_spatial.legend(loc="upper right")
    ax_spatial.grid(True, alpha=0.3)
    
    time_text = ax_spatial.text(0.05, 0.95, "", transform=ax_spatial.transAxes, fontsize=12)

    def update(frame):
        # 1. עדכון הגריד המרחבי
        line_p1.set_data(grid2d.x, marginals_1[frame])
        line_p2.set_data(grid2d.x, marginals_2[frame])
        time_text.set_text(f"Time $t = {time_array[frame]:.3f}$")
        
        # 2. עדכון ספירת בלוך
        b.clear()
        
        # וקטור לאלקטרון 1 (אדום)
        b.add_vectors(bloch_vectors_1[frame])
        
        # וקטור לאלקטרון 2 (כחול)
        b.add_vectors(bloch_vectors_2[frame])
        
        b.vector_color = ["#E63946", "#457B9D"]
        b.make_sphere()
        
        return line_p1, line_p2, time_text

    print("\nRendering animation...")
    anim = FuncAnimation(fig, update, frames=time_steps, interval=1000/fps, blit=False)
    plt.tight_layout()
    plt.show()
    return anim