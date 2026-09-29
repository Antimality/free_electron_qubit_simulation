from circuit_simulator import PARAMS, run_simulation
from visualizations_2d import animate_quantum_circuit_2d

if __name__ == "__main__":
    # SWAP should take |1, 0> to |0, 1>
    initial_state = [1, 0]
    target_state = [0, 1]

    # SWAP is composed of 3 alternating CNOTs:
    # 1. CNOT(Control Q2, Target Q1) -> "H1", "CZ", "H1"
    # 2. CNOT(Control Q1, Target Q2) -> "H2", "CZ", "H2"
    # 3. CNOT(Control Q2, Target Q1) -> "H1", "CZ", "H1"
    circuit_sequence = [
        "HOLD",
        "H1", "CZ", "H1", "HOLD",  # 1st CNOT
        "H2", "CZ", "H2", "HOLD",  # 2nd CNOT
        "H1", "CZ", "H1", "HOLD"   # 3rd CNOT
    ]  # fmt: off

    res, grid2d, gate_schedule = run_simulation(
        PARAMS, initial_state, target_state, circuit_sequence
    )

    print("Launching Integrated Circuit Dashboard...")
    anim, controls = animate_quantum_circuit_2d(
        grid2d,
        res["time"],
        res["m1"],
        res["m2"],
        res["b1"],
        res["b2"],
        res["fid_q1"],
        res["fid_q2"],
        center_q1=-PARAMS["D"] / 2.0,
        center_q2=PARAMS["D"] / 2.0,
        mu=PARAMS["mu"],
        sigma=PARAMS["sigma_well"],
        D=PARAMS["D"],
        gate_schedule=gate_schedule,
        initial_state=initial_state,
        target_state=target_state,
        fps=PARAMS["fps"],
    )
