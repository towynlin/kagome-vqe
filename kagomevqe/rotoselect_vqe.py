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
        callback: Callable[[int, int, int, str, np.ndarray, float], None] | None = None,
    ) -> None:
        """Mostly the same as VQE.
        There's no optimizer to pass.
        The callback has 2 count integers instead of 1.
        The first argument is callback_count.
        The second is run_count, the number of times estimator.run has been called,
        including retries after catching an exception.
        The third int is the index of the parameterized gate that was optimized.
        The fourth arg indicates which gate was selected, "rx", "ry", or "rz".
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
        minimized_energy = 99999
        greatest_allowed_increase = 99999
        greatest_allowed_decrease = -99999
        updates_skipped = 0
        should_stop = False
        while not should_stop:
            greatest_increase = 0
            greatest_decrease = 0
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

                estimator_result = None

                try:
                    job = self.estimator.run(
                        circuits=circuits,
                        observables=[operator] * batch_size,
                        parameter_values=parameters,
                    )
                    run_count += 1
                except Exception as exc:
                    print(f"Caught exception calling estimator.run: {exc}")
                    print(f"Retrying once...")
                    job = self.estimator.run(
                        circuits=circuits,
                        observables=[operator] * batch_size,
                        parameter_values=parameters,
                    )
                    run_count += 1

                try:
                    estimator_result = job.result()
                except Exception as exc:
                    print(f"Caught exception calling job.result: {exc}")
                    print(f"Retrying the whole job once more...")
                    job = self.estimator.run(
                        circuits=circuits,
                        observables=[operator] * batch_size,
                        parameter_values=parameters,
                    )
                    run_count += 1
                    estimator_result = job.result()

                assert isinstance(estimator_result, EstimatorResult)
                energies = estimator_result.values
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
                print(f"C = {C}")
                y = np.array(
                    [
                        2 * energies[0] - energies[1] - energies[2],
                        2 * energies[0] - energies[3] - energies[4],
                        2 * energies[0] - energies[5] - energies[6],
                    ]
                )
                print(f"y = {y}")
                x = np.array(
                    [
                        energies[1] - energies[2],
                        energies[3] - energies[4],
                        energies[5] - energies[6],
                    ]
                )
                print(f"x = {x}")
                if np.all(x != 0):
                    print(f"y/x = {y/x}")
                B = np.arctan2(y, x)
                print(f"B = {B}")
                A = np.sqrt(np.square(y) + np.square(x)) / 2
                print(f"A = {A}")
                minima = C - A
                print(f"minima = {minima}")

                # In case of multiple minima, argmin returns the first.
                # So we organize results in order rz, ry, rx
                # since rz is the native gate on guadalupe.
                best_gate = np.argmin(minima)
                print(f"best_gate = {best_gate}")

                gate_name = "same"
                delta = minima[best_gate] - minimized_energy
                if delta > greatest_allowed_increase or delta < greatest_allowed_decrease:
                    print(
                        f"Skipping update of gate {d} because it would significantly change the minimized energy"
                    )
                    updates_skipped += 1
                    if updates_skipped >= D:
                        should_stop = True
                    new_theta = 𝜃[d + D * 7]
                else:
                    updates_skipped = 0
                    if delta > greatest_increase:
                        greatest_increase = delta
                    if delta < greatest_decrease:
                        greatest_decrease = delta

                    gate_name = ["rz", "ry", "rx"][best_gate]
                    minimized_energy = minima[best_gate]
                    new_theta = -HALF_PI - B[best_gate]
                    if new_theta <= -np.pi:
                        new_theta += 2 * np.pi

                    gate_choices = [RZGate, RYGate, RXGate]
                    self._roto_trans.replacement_gate = gate_choices[best_gate]
                    self._roto_trans.parameter_index = d
                    self.ansatz = self._roto_trans(self.ansatz)

                print(f"new_theta = {new_theta}")
                indices_to_update = np.array(d + D * np.array(range(batch_size)))
                𝜃[indices_to_update] = new_theta

                if self.callback is not None:
                    callback_count += 1
                    self.callback(
                        callback_count,
                        run_count,
                        d,
                        gate_name,
                        𝜃[:D],
                        minimized_energy,
                    )

            print(
                f"Updating greatest allowed increase from {greatest_allowed_increase} to {greatest_increase}"
            )
            greatest_allowed_increase = greatest_increase
            print(
                f"Updating greatest allowed decrease from {greatest_allowed_decrease} to {greatest_decrease}"
            )
            greatest_allowed_decrease = greatest_decrease

            if relative_error(minimized_energy) < 0.0001:
                should_stop = True

        optimizer_time = time() - start_time
        optimizer_result = OptimizerResult()
        optimizer_result.fun = minimized_energy
        optimizer_result.x = 𝜃[:D]

        return self._build_vqe_result(self.ansatz, optimizer_result, [], optimizer_time)

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
