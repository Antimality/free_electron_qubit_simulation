import time
import numpy as np

# ייבוא המחלקות שיצרנו
from grid import Grid2D
from potentials import double_well_2d, coulomb_interaction, complex_absorbing_potential_2d
from wavefunction import Wavefunction2D
from propagator import SplitOperatorPropagator2D
from visualizations_2d import animate_2qubit_simulation

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. הגדרת פרמטרים וקבועים
    # ---------------------------------------------------------
    L_grid = 1.0
    N_grid = 256  # מותאם לדו-מימד כדי לשמור על ביצועים (256x256 = 65k נקודות)
    m = 1.0
    hbar = 1.0
    
    # פרמטרי הבור הכפול
    mu = 0.4        # מרחק הבורות מהמרכז
    sigma = 0.12    # רוחב הבורות
    U0 = -300.0     # עומק הבורות

    # עוצמת אינטראקציית קולון (ניתן לשחק עם זה כדי להאיץ/להאט את השזירה)
    k_e = 2.0      
    
    # פרמטרי זמן
    total_time = 10.0
    dt_anim = 0.1   # קצב עדכון הפריים באנימציה

    # ---------------------------------------------------------
    # 2. בניית המרחב והפוטנציאלים
    # ---------------------------------------------------------
    print("Setting up 2D Configuration Space...")
    grid2d = Grid2D(L_grid, N_grid, padding_mult=7.0)
    
    # הרכבת הפוטנציאל הכולל (בורות + קולון + בליעה מדומה בקצוות)
    V_wells = double_well_2d(grid2d, U0=U0, sigma=sigma, mu=mu)
    V_coulomb = coulomb_interaction(grid2d, k_e=k_e, softening_a=0.1)
    abso = complex_absorbing_potential_2d(grid2d, width=0.2, strength=100.0)
    
    potential_2d = V_wells + V_coulomb - 1j * abso

    # ---------------------------------------------------------
    # 3. אתחול פונקציית הגל לשני חלקיקים
    # ---------------------------------------------------------
    print("Initializing 2D Wavefunction...")
    wf_initial = Wavefunction2D(grid2d)
    
    # אנחנו מציבים את אלקטרון 1 בבור השמאלי (-mu) ואלקטרון 2 בבור הימני (+mu)
    # זה מקביל למצב הלוגי |01>
    wf_initial.init_gaussian(mu1=-mu, mu2=mu, sigma=sigma)

    # ---------------------------------------------------------
    # 4. אתחול מנוע ה-Split-Operator הדו-ממדי
    # ---------------------------------------------------------
    print("Initializing Split-Operator 2D Engine...")
    prop2d = SplitOperatorPropagator2D(grid2d, potential_2d, m=m, hbar=hbar)

    # ---------------------------------------------------------
    # 5. הפעלת האנימציה (קידום וויזואליזציה)
    # ---------------------------------------------------------
    print("Starting simulation and animation...")
    anim = animate_2qubit_simulation(
        grid2d=grid2d,
        prop2d=prop2d,
        wf_initial=wf_initial,
        total_time=total_time,
        dt_anim=dt_anim,
        mu=mu,
        fps=30
    )