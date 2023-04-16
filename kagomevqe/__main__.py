from argparse import ArgumentParser
from kagomevqe import (
    GuadalupeExpressibleJosephsonSampler,
    GuadalupeKagomeExtended16,
    GuadalupeKagomeRotationalSymmetry,
    KagomeHamiltonian,
    Kagome16AsymmetricHamiltonian,
    RetryEstimator,
    RotoselectRepository,
    RotoselectVQE,
    Rotosolve,
    SimpleRepository,
)
import matplotlib.pyplot as plt
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.primitives import (
    BackendEstimator,
    BaseEstimator,
    Estimator as LocalEstimator,
)
from qiskit.providers.fake_provider import FakeGuadalupeV2
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options
from qiskit_ibm_runtime.options import EnvironmentOptions, SimulatorOptions
from qiskit_ionq import IonQProvider
from qiskit_ionq.ionq_backend import IonQBackend
from time import time, strftime
from typing import Any


parser = ArgumentParser("python -m kagomevqe")
parser.add_argument(
    "-b",
    "--backend",
    default="local",
    choices=["local", "simulator", "guadalupe", "ionq"],
    help="Where to run. Default: local.",
)
parser.add_argument(
    "-l",
    "--lattice",
    default="kagome-unit",
    choices=["kagome-unit", "asymmetric"],
    help="The Hamiltonian observable. Default: kagome-unit.",
)
parser.add_argument(
    "-a",
    "--ansatz",
    default="josephson",
    choices=["josephson", "rotsym", "combo"],
    help="The parameterized circuit. Default: josephson.",
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="rotoselect",
    choices=["rotoselect", "rotosolve", "bfgs"],
    help="Variational optimization strategy. Default: rotoselect.",
)
parser.add_argument(
    "-m",
    "--maxiter",
    type=int,
    default=100,
    help="Maximum optimizer iterations. Default: 100.",
)
parser.add_argument(
    "--no-noise",
    action="store_true",
    help="Don't add noise to simulations.",
)
args = parser.parse_args()


options = Options()
options.environment = EnvironmentOptions(log_level="DEBUG")
options.resilience_level = 2
options.optimization_level = 3

LOCAL = False
IONQ = False
backend = ""
if args.backend == "local":
    LOCAL = True
    print("Running locally")
elif args.backend == "simulator":
    backend = "ibmq_qasm_simulator"
    print("Running on the IBM QASM simulator")
elif args.backend == "guadalupe":
    backend = "ibmq_guadalupe"
    print("Running on IBM Guadalupe")
elif args.backend == "ionq":
    IONQ = True
    print("Running on the IonQ simulator")


reps = 2
variant = "original"
ham_class = KagomeHamiltonian
if args.lattice == "asymmetric":
    reps = 3
    variant = "fill16"
    ham_class = Kagome16AsymmetricHamiltonian
    print("Using asymmetric extended Kagome lattice Hamiltonian")

prepend_rotsym = args.ansatz == "combo"
ansatz = GuadalupeExpressibleJosephsonSampler(
    reps=reps, variant=variant, prepend_rotsym=prepend_rotsym
)

if args.ansatz == "rotsym":
    if args.lattice == "asymmetric":
        ansatz = GuadalupeKagomeExtended16()
        print("Using a simple Bell state ansatz on the extended lattice")
    else:
        ansatz = GuadalupeKagomeRotationalSymmetry()
        print("Using rotational symmetry ansatz")
else:
    combo = ""
    if prepend_rotsym:
        combo = " after initial rotational symmetry Bell state"
    print(f"Using highly expressible Josephson sampler ansatz{combo}")


if args.optimizer == "rotoselect":
    repo = RotoselectRepository(num_params=ansatz.num_parameters)
else:
    repo = SimpleRepository(num_params=ansatz.num_parameters)


hamiltonian = ham_class.pauli_sum_op()
gs_energy = ham_class.ground_state_energy()


def relative_error(val: float) -> float:
    return abs((gs_energy - val) / gs_energy)


def execute_timed(estimator: BaseEstimator, session: Session | None = None):
    t = strftime("%m/%d %H:%M:%S%z")
    print(f"{t} Starting")
    start = time()
    if args.optimizer == "rotoselect":
        assert isinstance(repo, RotoselectRepository)
        x0 = 0.1 * (np.random.rand(ansatz.num_parameters) - 0.5)
        vqe = RotoselectVQE(
            estimator=estimator,
            ansatz=ansatz,
            initial_point=x0,  # type: ignore
            repository=repo,
            maxiter=args.maxiter,
        )
    else:
        assert isinstance(repo, SimpleRepository)
        x0 = [-np.pi / 2] * ansatz.num_parameters
        if args.optimizer == "rotosolve":
            optimizer = Rotosolve(maxiter=args.maxiter)
        else:
            optimizer = L_BFGS_B(maxiter=args.maxiter)
        vqe = VQE(estimator, ansatz, optimizer, initial_point=x0, callback=repo.update)
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
    if args.no_noise:
        estimator = LocalEstimator()
    else:
        fake_guadalupe = FakeGuadalupeV2()
        noise_model = NoiseModel.from_backend(fake_guadalupe)
        estimator = AerEstimator(
            backend_options={
                "method": "density_matrix",
                "coupling_map": fake_guadalupe.coupling_map,
                "noise_model": noise_model,
            },
            run_options={"shots": 1024},
        )
        print("Applying guadalupe noise model")
    execute_timed(estimator)
elif IONQ:
    backend = IonQProvider().get_backend("ionq_simulator")
    assert isinstance(backend, IonQBackend)
    if not args.no_noise:
        print("Setting noise model to aria-1")
        backend.set_options(noise_model="aria-1")
    execute_timed(BackendEstimator(backend))
else:
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance="ibm-q-community/ibmquantumawards/open-science-22",
    )
    with Session(service=service, backend=backend) as session:
        if args.backend == "simulator" and not args.no_noise:
            fake_guadalupe = FakeGuadalupeV2()
            noise_model = NoiseModel.from_backend(fake_guadalupe)
            cmlist = [[a, b] for a, b in fake_guadalupe.coupling_map]
            options.simulator = SimulatorOptions(
                noise_model=noise_model,
                coupling_map=cmlist,
                basis_gates=fake_guadalupe.operation_names,
            )
            print("Applying guadalupe noise model")
        if args.optimizer == "rotoselect":
            # Retry is handled in RotoselectVQE
            estimator = Estimator(session=session, options=options)
        else:
            estimator = RetryEstimator(session=session, options=options)
        execute_timed(estimator, session)

t = strftime("%m-%d-%H-%M-%S%z")
np.save(f"data/{t}-values", repo.values)
np.save(f"data/{t}-params", repo.parameters)
np.save(f"data/{t}-gates", repo.gate_names)

plt.rcParams.update({"font.size": 16})  # enlarge matplotlib fonts

plt.clf()
if args.optimizer == "bfgs":
    label = "VQE"
    xlabel = "Iterations"
elif args.optimizer == "rotoselect":
    label = "RotoselectVQE"
    xlabel = "Iterations (gates optimized)"
else:
    label = "Rotosolve"
    xlabel = "Iterations (circuit runs, 4 per gate)"
plt.plot(repo.values, color="purple", lw=2, label=label)
plt.ylabel("Energy")
plt.xlabel(xlabel)
plt.axhline(y=gs_energy, color="tab:red", ls="--", lw=2, label=f"Target: {gs_energy}")
plt.legend()
plt.grid()
plt.savefig(f"data/{t}-fig-values")

plt.clf()
plt.hist(repo.values, bins=40, density=True)
plt.ylabel("Density")
plt.xlabel("Energy")
plt.grid()
plt.savefig(f"data/{t}-fig-hist-density")
