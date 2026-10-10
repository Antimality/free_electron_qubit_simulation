from circuit_simulator import PARAMS, run_simulation
from visualizations_2d import animate_quantum_circuit_2d

if __name__ == "__main__":
    # CNOT should take 1,0 to 1,0
    initial_state = [1, 0]
    target_state = [1, 0]
    circuit_sequence = ["HOLD", "H1", "HOLD", "CZ", "HOLD", "H1", "HOLD"]

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
