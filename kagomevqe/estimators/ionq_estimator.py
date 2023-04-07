from itertools import accumulate
from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimator, EstimatorResult
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.tools.monitor import job_monitor
from qiskit_ionq.ionq_backend import IonQBackend
from typing import Sequence


def _run_circuits(
    circuits: QuantumCircuit | list[QuantumCircuit],
    backend: IonQBackend,
    monitor: bool = False,
    **run_options,
) -> tuple[Result, list[dict]]:
    """Remove metadata of circuits and run the circuits on a backend.
    Args:
        circuits: The circuits
        backend: The backend
        monitor: Enable job minotor if True
        **run_options: run_options
    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        circ.metadata = {}

    results = []
    first_job_id = ""
    success = True
    for circ in circuits:
        job = backend.run(circ, **run_options)
        if first_job_id == "":
            first_job_id = job.job_id()
        if monitor:
            job_monitor(job)

        exp_result = None
        ionq_job_result = job.result()
        if ionq_job_result is None:
            success = False
        else:
            success = success and ionq_job_result.success
            data = ExperimentResultData(counts=ionq_job_result.get_counts())
            exp_result = ExperimentResult(
                shots=1024,
                success=ionq_job_result.success,
                data=data,
            )
        results.append(exp_result)

    multiresult = Result(
        backend.name,
        backend.version,
        qobj_id=first_job_id,
        job_id=first_job_id,
        success=success,
        results=results,
    )

    return multiresult, metadata


class IonQEstimator(BackendEstimator):
    def __init__(self, backend: IonQBackend, options: dict | None = None):
        super().__init__(backend, options=options)
        self._backend = backend

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        # Transpile
        self._grouping = list(zip(circuits, observables))
        transpiled_circuits = self.transpiled_circuits
        num_observables = [len(m) for (_, m) in self.preprocessed_circuits]
        accum = [0] + list(accumulate(num_observables))

        # Bind parameters
        parameter_dicts = [
            dict(zip(self._parameters[i], value))
            for i, value in zip(circuits, parameter_values)
        ]
        bound_circuits = [
            transpiled_circuits[circuit_index]
            if len(p) == 0
            else transpiled_circuits[circuit_index].bind_parameters(p)
            for i, (p, n) in enumerate(zip(parameter_dicts, num_observables))
            for circuit_index in range(accum[i], accum[i] + n)
        ]
        bound_circuits = self._bound_pass_manager_run(bound_circuits)

        # Run
        result, metadata = _run_circuits(bound_circuits, self._backend, **run_options)

        return self._postprocessing(result, accum, metadata)
