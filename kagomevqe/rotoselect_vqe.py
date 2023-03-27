from kagomevqe import RotoselectTranslator
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import VQE, VQEResult
from qiskit.algorithms.optimizers import OptimizerResult
from qiskit.circuit.library import RZGate, RYGate, RXGate
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from time import time
from typing import Callable, List, Sequence, Tuple


HALF_PI = np.pi / 2


class RotoselectVQE(VQE):
    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        initial_point: Sequence[float],
        callback: Callable[
            [int, int, Tuple[bool, str, str], List[str], np.ndarray, float], None
        ]
        | None = None,
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
        iteration = 0
        minimized_energy = 99999
        absolute_lowest_energy_so_far = minimized_energy
        absolute_lowest_params = 𝜃[:D]
        absolute_lowest_gates = []
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

                minima, B, best_gate = self._analyze_energies(energies)

                minimized_energy = minima[best_gate]
                new_theta = -HALF_PI - B[best_gate]
                if new_theta <= -np.pi:
                    new_theta += 2 * np.pi

                gate_types = [RZGate, RYGate, RXGate]
                self._roto_trans.replacement_gate = gate_types[best_gate]
                self._roto_trans.parameter_index = d
                self.ansatz = self._roto_trans(self.ansatz)
                did_change, _, _ = self._roto_trans.last_substitution
                same_different = "same"
                if did_change:
                    same_different = "different"

                print(
                    f"new_theta = {new_theta}, old_theta = {𝜃[d + D * 7]} ({same_different} gate)"
                )
                indices_to_update = np.array(d + D * np.array(range(batch_size)))
                𝜃[indices_to_update] = new_theta

                if minimized_energy < absolute_lowest_energy_so_far:
                    absolute_lowest_energy_so_far = minimized_energy
                    absolute_lowest_params = 𝜃[:D]
                    absolute_lowest_gates = (
                        self._roto_trans._last_parameterized_gate_name_list
                    )

                if self.callback is not None:
                    self.callback(
                        iteration,
                        d,
                        self._roto_trans.last_substitution,
                        self._roto_trans._last_parameterized_gate_name_list,
                        𝜃[:D],
                        minimized_energy,
                    )

            iteration += 1
            if iteration >= 1:
                print(f"\nBest so far:\n")
                print(absolute_lowest_energy_so_far)
                print(absolute_lowest_params)
                print(absolute_lowest_gates)
            if iteration >= 9:
                should_stop = True

        optimizer_time = time() - start_time
        optimizer_result = OptimizerResult()
        optimizer_result.fun = absolute_lowest_energy_so_far
        optimizer_result.x = absolute_lowest_params
        print(f"Final absolute lowest gates: {absolute_lowest_gates}")

        return self._build_vqe_result(self.ansatz, optimizer_result, [], optimizer_time)

    def _run_and_maybe_retry_estimator(
        self,
        circuits: List[QuantumCircuit],
        operators: Sequence[PauliSumOp],
        parameters: Sequence[Sequence[float]],
    ) -> np.ndarray:
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

        if not isinstance(estimator_result, EstimatorResult):
            print(f"Estimator failure. No result. Creating empty one.")
            estimator_result = EstimatorResult(np.array([]), [])

        return estimator_result.values

    def _analyze_energies(
        self, energies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
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

        # In case of multiple equal minima, argmin returns the first.
        # So we organize results in order of increasing transpiled depth:
        # [rz, ry, rx] since rz is the native rotation gate on guadalupe.
        best_gate = int(np.argmin(minima))

        return (minima, B, best_gate)

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
