from kagomevqe import (
    KagomeExpressibleJosephsonSampler,
    KagomeHamiltonian,
    Rotosolve,
    VQELog,
    relative_error,
)
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options
from qiskit_ibm_runtime.options import EnvironmentOptions, TranspilationOptions
import sys
from time import time, strftime


if len(sys.argv) < 2:
    print("You must provide one command line argument to")
    print("specify where to run the quantum circuits.")
    print("Valid options are: simulator, guadalupe")
    sys.exit(2)

options = Options()
options.environment = EnvironmentOptions(log_level="DEBUG")
# options.transpilation = TranspilationOptions(skip_transpilation=True)
options.resilience_level = 2
options.optimization_level = 3

if sys.argv[1] == "simulator":
    print("Running on the IBM statevector simulator")
    backend = "simulator_statevector"
elif sys.argv[1] == "guadalupe":
    backend = "ibmq_guadalupe"
    print("Running on IBM Guadalupe")
    options.resilience_level = 3
else:
    print(f"Invalid run location argument: {sys.argv[1]}")
    print("Valid options are: local, simulator, guadalupe")
    sys.exit(2)

log = VQELog()
ansatz = KagomeExpressibleJosephsonSampler()
hamiltonian = KagomeHamiltonian.pauli_sum_op()
x0 = 0.1 * (np.random.rand(ansatz.num_parameters) - 0.5)
optimizer = Rotosolve()
service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q-community/ibmquantumawards/open-science-22",
)
with Session(service=service, backend=backend) as session:
    t = strftime("%m/%d %H:%M:%S%z")
    print(f"{t} Starting session")
    start = time()
    estimator = Estimator(session=session, options=options)
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=x0,  # type: ignore
        callback=log.update,
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
        result = f"Interrupted.\nLast best value: {measured}\nLast params: {log.parameters[-1]}"

    end = time()
    print(f"Mitigated result for session {session.session_id}: {result}")
    print(f"Execution time (s): {end - start:.2f}")

    rel_error = relative_error(measured)
    print(f"Relative error: {rel_error}")
