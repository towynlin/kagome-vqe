from kagomevqe import (
    KagomeExpressibleJosephsonSampler,
    KagomeRotoselectShallow,
    KagomeHamiltonian,
    RotoselectVQE,
    VQELog,
    relative_error,
)
import matplotlib.pyplot as plt
import numpy as np
from qiskit.primitives import (
    BaseEstimator,
    Estimator as LocalEstimator,
)
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options
from qiskit_ibm_runtime.options import EnvironmentOptions, TranspilationOptions
import sys
from time import time, strftime


if len(sys.argv) < 2:
    print("You must provide one command line argument to")
    print("specify where to run the quantum circuits.")
    print("Valid options are: local, simulator, guadalupe")
    sys.exit(2)

options = Options()
options.environment = EnvironmentOptions(log_level="DEBUG")
# options.transpilation = TranspilationOptions(skip_transpilation=True)
options.resilience_level = 2
options.optimization_level = 3

LOCAL = False
backend = ""
if sys.argv[1] == "local":
    print("Running locally")
    LOCAL = True
elif sys.argv[1] == "simulator":
    print("Running on the IBM QASM simulator")
    backend = "ibmq_qasm_simulator"
elif sys.argv[1] == "guadalupe":
    backend = "ibmq_guadalupe"
    print("Running on IBM Guadalupe")
    # Mar 15 in slack Vishal said not recommended, can lead to memory errors
    # options.resilience_level = 3
else:
    print(f"Invalid run location argument: {sys.argv[1]}")
    print("Valid options are: local, simulator, guadalupe")
    sys.exit(2)

log = VQELog()
ansatz = KagomeExpressibleJosephsonSampler()
hamiltonian = KagomeHamiltonian.pauli_sum_op()
x0 = 0.1 * (np.random.rand(ansatz.num_parameters) - 0.5)


def execute_timed(estimator: BaseEstimator, session: Session | None = None):
    t = strftime("%m/%d %H:%M:%S%z")
    print(f"{t} Starting")
    start = time()
    vqe = RotoselectVQE(
        estimator=estimator,
        ansatz=ansatz,
        initial_point=x0,  # type: ignore
        callback=log.rotoselect_update,
    )
    try:
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        if result.eigenvalue is not None:
            measured = result.eigenvalue.real
        else:
            measured = log.values[-1]
    except Exception as exc:
        print(f"\nException: {exc}\n")
        measured = log.values[-1]
        result = f"Exception.\nLast best value: {measured}\nLast params: {log.parameters[-1]}"
    except KeyboardInterrupt as exc:
        measured = log.values[-1]
        result = f"Interrupted.\nLast best value: {measured}\nLast params: {log.parameters[-1]}"

    end = time()
    session_str = ""
    if session is not None:
        session_str = f" for session {session.session_id}"
    print(f"Result{session_str}: {result}")
    print(f"Execution time (s): {end - start:.2f}")

    rel_error = relative_error(measured)
    print(f"Relative error: {rel_error}")


if LOCAL:
    execute_timed(LocalEstimator())
else:
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance="ibm-q-community/ibmquantumawards/open-science-22",
    )
    with Session(service=service, backend=backend) as session:
        estimator = Estimator(session=session, options=options)
        execute_timed(estimator, session)

plt.rcParams.update({"font.size": 16})  # enlarge matplotlib fonts

prefix = strftime("data/fig-%m-%d-%H-%M-%S%z")
plt.plot(log.values, color="purple", lw=2, label="RotoselectVQE")
plt.ylabel("Energy")
plt.xlabel("Iterations (gates optimized)")
plt.axhline(y=-18.0, color="tab:red", ls="--", lw=2, label="Target: -18.0")
plt.legend()
plt.grid()
plt.savefig(f"{prefix}-values")
plt.clf()

plt.hist(log.values)
plt.ylabel("Number of points")
plt.xlabel("Energy")
plt.grid()
plt.savefig(f"{prefix}-hist")
plt.clf()

plt.hist(log.values, density=True)
plt.ylabel("Density")
plt.xlabel("Energy")
plt.grid()
plt.savefig(f"{prefix}-hist-density")
