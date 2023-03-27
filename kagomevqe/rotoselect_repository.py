from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from time import time, strftime
from typing import Any, List, Tuple


@dataclass
class RotoselectRepository:
    first_time: float = 0.0
    last_time: float = 0.0
    values: np.ndarray = field(default=np.empty((0,), dtype=float))
    parameters: np.ndarray = field(default=np.empty((0, 96), dtype=float))
    gate_names: np.ndarray = field(default=np.empty((0, 96), dtype=str))

    def __post_init__(self):
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

        # Get the indices of the lowest five percent of values, unsorted
        lowest_5p_indices = np.argpartition(self.values, five_percent)[:five_percent]

        # Sort the indices so the minimum value's index is first, then ascending
        lowest_5p_indices = lowest_5p_indices[
            np.argsort(self.values[lowest_5p_indices])
        ]

        candidate_values = self.values[lowest_5p_indices]

        # Find the lowest value near enough to the median
        m = np.median(candidate_values)
        med_var = 0.5 * np.mean(candidate_values - m)
        outliers = candidate_values - m < med_var
        i = 0
        while outliers[i]:
            i += 1

        best_index = lowest_5p_indices[i]
        optimum = (
            self.values[best_index],
            self.parameters[best_index],
            self.gate_names[best_index],
        )
        return optimum  # type: ignore
