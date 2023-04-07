import numpy as np
from qiskit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_ibm_runtime import Estimator, RuntimeJob
from typing import Any, Sequence


class RetryEstimator(Estimator):
    def run(
        self,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[float] | Sequence[Sequence[float]] | None = None,
        **kwargs: Any,
    ) -> RuntimeJob:
        try:
            job = super().run(circuits, observables, parameter_values, **kwargs)
        except Exception as exc:
            print(f"Caught exception calling estimator.run: {exc}")
            print(f"Retrying once...")
            job = super().run(circuits, observables, parameter_values, **kwargs)

        return job
