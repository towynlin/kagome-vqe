from kagomevqe import RotoselectTranslator
from kagomevqe.vqelog import relative_error
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import VQE, VQEResult
from qiskit.algorithms.optimizers import OptimizerResult
from qiskit.circuit.library import RZGate, RYGate, RXGate
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from time import time
from typing import Any, Callable, List, Sequence


HALF_PI = np.pi / 2


class RotoselectVQE(VQE):
    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        initial_point: Sequence[float],
        callback: Callable[[int, int, str, np.ndarray, float], None] | None = None,
    ) -> None:
        """Mostly the same as VQE.
        There's no optimizer to pass.
        The first int arg is the zero-based index of the iteration.
        The second int arg is the zero-based index of the parameterized gate that was optimized.
        The third arg indicates which gate was selected, "rx", "ry", or "rz".
        We don't pass the metadata to the callback,
        and only the extrapolated minimized energy from each job is passed,
        not all the results of circuit evaluations.
        """
        self.estimator = estimator
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.callback = callback
        self._roto_trans = RotoselectTranslator()

    def compute_minimum_eigenvalue(
        self,
        operator: PauliSumOp,
    ) -> VQEResult:
        start_time = time()
        batch_size = 8
        x0 = np.array(self.initial_point)
        D = len(x0)
        𝜃 = np.tile(x0, batch_size)
        run_count = 0
        callback_count = 0
        iteration = 0
        minimized_energy = 99999
        should_stop = False
        while not should_stop:
            for d in range(D):
                circuits = self._get_circuit_structure_variants(d)
                assert len(circuits) == batch_size

                𝜃[d] = 0
                indices_plus = [d + D, d + 3 * D, d + 5 * D]
                𝜃[indices_plus] = HALF_PI
                indices_minus = [d + 2 * D, d + 4 * D, d + 6 * D]
                𝜃[indices_minus] = -HALF_PI

                # Reshape into [array, array, ...] with each element of len D
                parameters = np.reshape(𝜃, (-1, D)).tolist()
                assert len(parameters) == batch_size

                energies = self._run_and_maybe_retry_estimator(
                    circuits, [operator] * batch_size, parameters
                )
                print(f"energies = {energies}")

                C = (
                    np.array(
                        [
                            energies[1] + energies[2],
                            energies[3] + energies[4],
                            energies[5] + energies[6],
                        ]
                    )
                    / 2
                )
                y = np.array(
                    [
                        2 * energies[0] - energies[1] - energies[2],
                        2 * energies[0] - energies[3] - energies[4],
                        2 * energies[0] - energies[5] - energies[6],
                    ]
                )
                x = np.array(
                    [
                        energies[1] - energies[2],
                        energies[3] - energies[4],
                        energies[5] - energies[6],
                    ]
                )
                B = np.arctan2(y, x)
                A = np.sqrt(np.square(y) + np.square(x)) / 2
                minima = C - A
                # print(f"minima = {minima}")

                # In case of multiple equal minima, argmin returns the first.
                # So we organize results in order of increasing transpiled depth:
                # [rz, ry, rx] since rz is the native rotation gate on guadalupe.
                best_gate = np.argmin(minima)
                # print(f"best_gate = {best_gate}")

                gate_name = ["rz", "ry", "rx"][best_gate]
                minimized_energy = minima[best_gate]
                new_theta = -HALF_PI - B[best_gate]
                if new_theta <= -np.pi:
                    new_theta += 2 * np.pi

                gate_choices = [RZGate, RYGate, RXGate]
                self._roto_trans.replacement_gate = gate_choices[best_gate]
                self._roto_trans.parameter_index = d
                self.ansatz = self._roto_trans(self.ansatz)

                print(
                    f"new_theta = {new_theta}, old_theta = {𝜃[d + D * 7]} (maybe different gate)"
                )
                indices_to_update = np.array(d + D * np.array(range(batch_size)))
                𝜃[indices_to_update] = new_theta

                if self.callback is not None:
                    callback_count += 1
                    self.callback(
                        iteration,
                        d,
                        gate_name,
                        𝜃[:D],
                        minimized_energy,
                    )

            iteration += 1

        optimizer_time = time() - start_time
        optimizer_result = OptimizerResult()
        optimizer_result.fun = minimized_energy
        optimizer_result.x = 𝜃[:D]

        return self._build_vqe_result(self.ansatz, optimizer_result, [], optimizer_time)

    def _run_and_maybe_retry_estimator(
        self,
        circuits: List[QuantumCircuit],
        operators: Sequence[PauliSumOp],
        parameters: Sequence[Sequence[float]],
    ) -> np.ndarray:
        estimator_result = EstimatorResult(np.array([]), [])

        try:
            job = self.estimator.run(
                circuits=circuits,
                observables=operators,
                parameter_values=parameters,
            )
        except Exception as exc:
            print(f"Caught exception calling estimator.run: {exc}")
            print(f"Retrying once...")
            job = self.estimator.run(
                circuits=circuits,
                observables=operators,
                parameter_values=parameters,
            )

        try:
            estimator_result = job.result()
        except Exception as exc:
            print(f"Caught exception calling job.result: {exc}")
            print(f"Retrying the whole job once more...")
            job = self.estimator.run(
                circuits=circuits,
                observables=operators,
                parameter_values=parameters,
            )
            estimator_result = job.result()

        assert isinstance(estimator_result, EstimatorResult)
        return estimator_result.values

    def _get_circuit_structure_variants(self, d: int) -> List[QuantumCircuit]:
        # Set parameterized gate d to
        # 0,7: leave as is
        # 1,2: RZ
        # 3,4: RY
        # 5,6: RX
        self._roto_trans.parameter_index = d
        self._roto_trans.replacement_gate = RZGate
        circ_rz = self._roto_trans(self.ansatz)
        self._roto_trans.replacement_gate = RYGate
        circ_ry = self._roto_trans(self.ansatz)
        self._roto_trans.replacement_gate = RXGate
        circ_rx = self._roto_trans(self.ansatz)
        return [
            self.ansatz,
            circ_rz,
            circ_rz,
            circ_ry,
            circ_ry,
            circ_rx,
            circ_rx,
            self.ansatz,
        ]
