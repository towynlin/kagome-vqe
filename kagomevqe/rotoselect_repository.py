from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from time import time, strftime
from typing import Any, List, Tuple


@dataclass
class RotoselectRepository:
    parameters: np.ndarray = field()
    gate_names: np.ndarray = field()
    values: np.ndarray = field()
    first_time: float = 0.0
    last_time: float = 0.0

    def __init__(self, num_params: int):
        self.values = np.empty((0,), dtype=float)
        self.parameters = np.empty((0, num_params), dtype=float)
        self.gate_names = np.empty((0, num_params), dtype=str)
        self.first_time = time()

    def update(
        self,
        iteration: int,
        d: int,
        gate_change: Tuple[bool, str, str],
        gate_names: List[str],
        parameters: np.ndarray,
        energy: float,
    ):
        self.last_time = time()
        self.values = np.append(self.values, energy)
        self.parameters = np.append(self.parameters, [parameters], axis=0)
        self.gate_names = np.append(self.gate_names, [gate_names], axis=0)
        t = strftime("%m/%d %H:%M:%S%z")
        print(
            f"{t} Iteration {iteration} gate {d}: {gate_change}\tenergy: {energy: 012.08f}"
        )
        if self.values.size % 24 == 0:
            print(f"\nParameters: {parameters}\n")
            print(f"Gates: {gate_names}\n")

    def get_best_result(self) -> Tuple[float, np.ndarray, np.ndarray]:
        five_percent = int(0.05 * self.values.size)
        if five_percent < 1:
            return (0.0, np.array([]), np.array([]))

        # Get the indices of the lowest five percent of values, unsorted
        lowest_5p_indices = np.argpartition(self.values, five_percent)[:five_percent]

        # Sort the indices so the minimum value's index is first, then ascending
        lowest_5p_indices = lowest_5p_indices[
            np.argsort(self.values[lowest_5p_indices])
        ]

        # Remove data points preceded by large drop and succeeded by large increase
        to_delete = []
        for i, orig_idx in enumerate(lowest_5p_indices):
            # sanity check since we need the following data point
            if orig_idx < self.values.size - 1:
                orig_d1 = self.values[orig_idx] - self.values[orig_idx - 1]
                orig_d2 = self.values[orig_idx + 1] - self.values[orig_idx]
                if orig_d1 < -0.05 and orig_d2 > 0.05:
                    to_delete.append(i)

        lowest_5p_indices = np.delete(lowest_5p_indices, to_delete)

        best_index = lowest_5p_indices[0]

        optimum = (
            self.values[best_index],
            self.parameters[best_index],
            self.gate_names[best_index],
        )
        return optimum  # type: ignore
