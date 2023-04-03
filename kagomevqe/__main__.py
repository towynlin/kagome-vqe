from kagomevqe import (
    GuadalupeEfficientSU2,
    GuadalupeExpressibleJosephsonSampler,
    GuadalupeKagomeRotationalSymmetry,
    IonQEstimator,
    KagomeHamiltonian,
    RotoselectRepository,
    RotoselectVQE,
)
import matplotlib.pyplot as plt
import numpy as np
from qiskit.primitives import (
    BaseEstimator,
    Estimator as LocalEstimator,
)
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options
from qiskit_ibm_runtime.options import EnvironmentOptions
from qiskit_ionq import IonQProvider
from qiskit_ionq.ionq_backend import IonQBackend
import sys
from time import time, strftime


if len(sys.argv) < 2:
    print("You must provide one command line argument to")
    print("specify where to run the quantum circuits.")
    print("Valid options are: local, simulator, guadalupe, ionq")
    sys.exit(2)

options = Options()
options.environment = EnvironmentOptions(log_level="DEBUG")
options.resilience_level = 2
options.optimization_level = 3

LOCAL = False
IONQ = False
backend = ""
if sys.argv[1] == "local":
    LOCAL = True
    print("Running locally")
elif sys.argv[1] == "simulator":
    backend = "ibmq_qasm_simulator"
    print("Running on the IBM QASM simulator")
elif sys.argv[1] == "guadalupe":
    backend = "ibmq_guadalupe"
    print("Running on IBM Guadalupe")
elif sys.argv[1] == "ionq":
    IONQ = True
    print("Running on the IonQ simulator")
else:
    print(f"Invalid run location argument: {sys.argv[1]}")
    print("Valid options are: local, simulator, guadalupe")
    sys.exit(2)


ansatz = GuadalupeExpressibleJosephsonSampler()

if len(sys.argv) >= 3:
    if sys.argv[2] == "rotsym":
        ansatz = GuadalupeKagomeRotationalSymmetry()
        print("Using rotational symmetry ansatz")
    elif sys.argv[2] == "effsu2":
        ansatz = GuadalupeEfficientSU2()
        print("Using efficient SU2 ansatz")
    else:
        print("Using highly expressible Josephson sampler ansatz")

repo = RotoselectRepository(num_params=ansatz.num_parameters)
hamiltonian = KagomeHamiltonian.pauli_sum_op()
x0 = 0.1 * (np.random.rand(ansatz.num_parameters) - 0.5)


def relative_error(val: float) -> float:
    return abs((-18.0 - val) / -18.0)


def execute_timed(estimator: BaseEstimator, session: Session | None = None):
    t = strftime("%m/%d %H:%M:%S%z")
    print(f"{t} Starting")
    start = time()
    vqe = RotoselectVQE(
        estimator=estimator,
        ansatz=ansatz,
        initial_point=x0,  # type: ignore
        repository=repo,
    )
    try:
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        if result.eigenvalue is not None:
            measured = result.eigenvalue.real
            fname = strftime("data/%m-%d-%H-%M-%S%z-fig-ansatz")
            result.optimal_circuit.draw("mpl", filename=fname)
        else:
            measured = repo.get_best_result()[0]
    except Exception as exc:
        print(f"\nException: {exc}\n")
        measured, params, gates = repo.get_best_result()
        result = f"Exception.\nBest value: {measured}\nBest params: {params}\nBest gates: {gates}"
    except KeyboardInterrupt as exc:
        measured, params, gates = repo.get_best_result()
        result = f"Interrupted.\nBest value: {measured}\nBest params: {params}\nBest gates: {gates}"

    end = time()
    session_str = ""
    if session is not None:
        session_str = f" for session {session.session_id}"
    print(f"Result{session_str}: {result}")
    print(f"Execution time (s): {end - start:.2f}")

    rel_error = relative_error(measured)
    print(f"Relative error: {rel_error}")

    t = strftime("%m/%d %H:%M:%S%z")
    print(f"{t} Finished")


if LOCAL:
    execute_timed(LocalEstimator())
elif IONQ:
    backend = IonQProvider().get_backend("ionq_simulator")
    assert isinstance(backend, IonQBackend)
    execute_timed(IonQEstimator(backend))
else:
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance="ibm-q-community/ibmquantumawards/open-science-22",
    )
    with Session(service=service, backend=backend) as session:
        estimator = Estimator(session=session, options=options)
        execute_timed(estimator, session)

t = strftime("%m-%d-%H-%M-%S%z")
np.save(f"data/{t}-values", repo.values)
np.save(f"data/{t}-params", repo.parameters)
np.save(f"data/{t}-gates", repo.gate_names)

plt.rcParams.update({"font.size": 16})  # enlarge matplotlib fonts

plt.clf()
plt.plot(repo.values, color="purple", lw=2, label="RotoselectVQE")
plt.ylabel("Energy")
plt.xlabel("Iterations (gates optimized)")
plt.axhline(y=-18.0, color="tab:red", ls="--", lw=2, label="Target: -18.0")
plt.legend()
plt.grid()
plt.savefig(f"data/{t}-fig-values")

plt.clf()
plt.hist(repo.values, bins=30, density=True)
plt.ylabel("Density")
plt.xlabel("Energy")
plt.grid()
plt.savefig(f"data/{t}-fig-hist-density")
