from circuit_simulator import PARAMS, run_simulation
from visualizations_2d import animate_quantum_circuit_2d

if __name__ == "__main__":
    # Grover's Algorithm for 2 Qubits (Searching for |11>)
    initial_state = [0, 0]
    target_state = [1, 1]

    circuit_sequence = [
        "HH", "HOLD",  # Initialization
        "CZ", "HOLD",  # Oracle
        "ZZ", "HOLD",  # Diffuser
        "HH", "HOLD",
        "CZ", "HOLD",
        "HH", "HOLD",
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
